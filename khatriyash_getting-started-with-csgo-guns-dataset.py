import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
my_data_dir = '../input/csgo-guns-dataset'
os.listdir(my_data_dir)
os.listdir(my_data_dir+'/Train/AK-47')[0]
ak_img_dir = my_data_dir+'/Train/AK-47'+'/weapon_ak47.a320f13fea4f21d1eb3b46678d6b12e97cbd1052.jpg'
ak_img = imread(ak_img_dir)
plt.imshow(ak_img)
ak_img.shape
img_shape = (384, 512, 3)
ak_img.max()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range=20,
                               width_shift_range=0.10,
                               height_shift_range=0.10,
                               rescale=1/255,
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               fill_mode='nearest'
                              )
plt.imshow(ak_img)
plt.imshow(image_gen.random_transform(ak_img))
test_path = my_data_dir+'/Test/'
train_path = my_data_dir+'/Train/'
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
batch_size=16
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=img_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='categorical')
test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=img_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='categorical',shuffle=False)
train_image_gen.class_indices