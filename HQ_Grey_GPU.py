import numpy as np
import tensorflow as tf
import time
from PIL import ImageGrab
from PIL import Image

def convert_text_to_image(file_path):
    # 读取文本文件内容
    with open(file_path, 'r') as file:
        content = file.readlines()

    # 提取灰度值
    pixels = []
    for line in content:
        line = line.strip()  # 去除行首尾的空格和换行符
        row = [int(val) for val in line.split()]  # 将每个灰度值转换为整数
        pixels.append(row)

    # 创建图像对象
    image = Image.new('L', (len(pixels[0]), len(pixels)))  # 创建单通道灰度图像

    # 设置图像像素
    for y, row in enumerate(pixels):
        for x, val in enumerate(row):
            image.putpixel((x, y), val)

    # 保存图像
    image.save('output_image.png')

    print("图像已保存为output_image.png")

def capture_screen():
    # 获取屏幕截图
    screen = ImageGrab.grab()
    return screen

def convert_pixels_to_text(screen):
    # 将屏幕像素转换为256灰度值
    pixels = np.array(screen.convert("L"))

    # 在GPU上执行灰度转换
    with tf.device('/GPU:0'):
        gray_pixels = tf.convert_to_tensor(pixels, dtype=tf.float32)
        gray_pixels = tf.expand_dims(gray_pixels, axis=0)
        gray_pixels = tf.expand_dims(gray_pixels, axis=-1)
        gray_pixels = tf.image.resize(gray_pixels, [720, 1280]) #抓取的分辨率
        gray_pixels = tf.squeeze(gray_pixels)
        gray_pixels = tf.round(gray_pixels)

    # 将灰度值转换为文本
    pixel_text = ""
    for row in gray_pixels.numpy():
        pixel_text += " ".join([str(int(value)) for value in row]) + "\n"

    return pixel_text

def main():
    # 捕获屏幕截图
    screen = capture_screen()

    # 将屏幕像素转换为文本（通过GPU加速）
    pixel_text = convert_pixels_to_text(screen)

    # 保存像素值文本文件
    with open("pixel_text.txt", "w") as file:
        file.write(pixel_text)


    print("像素文本已保存为文本文件")

if __name__ == "__main__":
    while(1):
        main()
        file_path = 'pixel_text.txt'  # 灰度值的文本文件路径
        convert_text_to_image(file_path)
        time.sleep(0.01) #改这个更改帧数之间的抓取时间

