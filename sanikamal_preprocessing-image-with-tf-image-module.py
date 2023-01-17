import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import tensorflow as tf

import os

matplotlib.rcParams['figure.figsize'] = (12.0, 8.0)

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
dir(tf.image)
image=plt.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_34.jpg')

image2=plt.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_150.jpg')

image3=plt.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_753.jpg')

image4=plt.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Test_1272.jpg')

image5=plt.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_778.jpg')

image6=plt.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Test_904.jpg')

plt.imshow(image)
image_adjust_the_brightness_1=tf.image.adjust_brightness(image,0.4)

image_adjust_the_brightness_2=tf.image.adjust_brightness(image,0.2)
plt.imshow(image_adjust_the_brightness_1)
plt.imshow(image_adjust_the_brightness_2)
image_adjust_the_contrast_1=tf.image.adjust_contrast(image,6)

image_adjust_the_contrast_2=tf.image.adjust_contrast(image,0.6)
plt.imshow(image_adjust_the_contrast_1)
plt.imshow(image_adjust_the_contrast_2)
image_adjust_gamma=tf.image.adjust_gamma(image,gamma=3,gain=2)
plt.imshow(image_adjust_gamma)
image_adjust_hue_1=tf.image.adjust_hue(image,0.5,name=None)

image_adjust_hue_2=tf.image.adjust_hue(image,-0.5,name=None)

image_adjust_hue_3=tf.image.adjust_hue(image,0.7,name=None)
plt.imshow(image_adjust_hue_1)
plt.imshow(image_adjust_hue_2)
plt.imshow(image_adjust_hue_3)
image_adjust_saturation_1=tf.image.adjust_saturation(image,0.2,name=None)

image_adjust_saturation_2=tf.image.adjust_saturation(image,0.5,name=None)

image_adjust_saturation_3=tf.image.adjust_saturation(image,0.9,name=None)
plt.imshow(image_adjust_saturation_1)
plt.imshow(image_adjust_saturation_2)
plt.imshow(image_adjust_saturation_3)
image_central_crop_1=tf.image.central_crop(image,0.2)

image_central_crop_2=tf.image.central_crop(image,0.5)

image_central_crop_3=tf.image.central_crop(image,0.9)
plt.imshow(image_central_crop_1)
plt.imshow(image_central_crop_2)
plt.imshow(image_central_crop_3)
image_crop_to_bounding_box_1=tf.image.crop_to_bounding_box(image, 10, 20, 670, 700)

image_crop_to_bounding_box_2=tf.image.crop_to_bounding_box(image, 2, 2, 750, 750)

image_crop_to_bounding_box_3=tf.image.crop_to_bounding_box(image, 40, 40, 1000, 1000)
plt.imshow(image_crop_to_bounding_box_1)
plt.imshow(image_crop_to_bounding_box_2)
plt.imshow(image_crop_to_bounding_box_3)
img_flip_left_right=tf.image.flip_left_right(image)

plt.imshow(img_flip_left_right)
img_flip_up_down=tf.image.flip_up_down(image)

plt.imshow(img_flip_up_down)
img_grayscale=tf.image.rgb_to_grayscale(image3)

print(img_grayscale.shape)
img_grayscale=tf.squeeze(img_grayscale)

print(img_grayscale.shape)
plt.imshow(img_grayscale,cmap='gray')
img_rot90=tf.image.rot90(image5, k=1, name=None)

plt.imshow(img_rot90)
img_transpose=tf.image.transpose(image4, name=None)

plt.imshow(img_transpose)

plt.show()