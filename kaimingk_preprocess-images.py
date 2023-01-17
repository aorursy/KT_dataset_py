import os



import cv2

import numpy as np

import pandas as pd

import keras

from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt
IMAGE_SHAPE = (224, 224, 3)

IMAGE_SIZE = (224, 224)



def preprocess_image(image):

    

    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 30), -4, 128)

    

    height, width, _ = image.shape

    center_x = int(width / 2)

    center_y = int(height / 2)

    radius = min(center_x, center_y)

    

    circle_mask = np.zeros((height, width), np.uint8)

    cv2.circle(circle_mask, (center_x, center_y), radius, color=1, thickness=-1)

    image = cv2.resize(cv2.bitwise_and(image, image, mask=circle_mask)[center_y - radius:center_y + radius, center_x - radius:center_x + radius], IMAGE_SIZE)

    

    return image
os.mkdir("/kaggle/processed")
os.mkdir("/kaggle/processed/train_images")

os.mkdir("/kaggle/processed/test_images")
train_path = "../input/aptos2019-blindness-detection/train_images"

test_path = "../input/aptos2019-blindness-detection/test_images"

dst_train_path = "/kaggle/processed/train_images"

dst_test_path = "/kaggle/processed/test_images"



for image_name in os.listdir(train_path):

    image_path = os.path.join(train_path, image_name)

    image = cv2.imread(image_path)

    image = preprocess_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    cv2.imwrite(os.path.join(dst_train_path, image_name), image)



for image_name in os.listdir(test_path):

    image_path = os.path.join(test_path, image_name)

    image = cv2.imread(image_path)

    image = preprocess_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    cv2.imwrite(os.path.join(dst_test_path, image_name), image)
os.listdir(dst_train_path)
sample = cv2.imread(os.path.join(dst_train_path, '0c43c79e8cfb.png'))

plt.imshow(sample)