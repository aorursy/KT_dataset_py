import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import os

import json

from sklearn import preprocessing

# Input data files are available in the "../input/" directory.
# This is not an optimal way to do it.

dataset_path = '/kaggle/input/pku-autonomous-driving/'

sample_submission_path = dataset_path + 'sample_submission.csv'

test_images_path = dataset_path + 'test_images/'

test_masks_path = dataset_path + 'test_masks/'

train_images_path = dataset_path + 'train_images/'

train_masks_path = dataset_path + 'train_masks/'

train_path = dataset_path + 'train.csv'

car_models_path = dataset_path + 'car_models/'

car_models_json_path = dataset_path + 'car_models_json/'
ss_df = pd.read_csv(sample_submission_path)

print(ss_df.head(n=5))

print(ss_df.shape)
train_df = pd.read_csv(train_path)

print(train_df.head(n=5))

print(train_df.shape)
# print train images in row 1

f, axarr = plt.subplots(2,3, figsize = (15,5))

for i in range(3):

    image_name = str(train_df['ImageId'].iloc[i])+'.jpg'

    im_read = cv2.imread(train_images_path + image_name,1)

    im = cv2.cvtColor(im_read, cv2.COLOR_BGR2RGB)

    axarr[0][i].imshow(im)

    axarr[0][i].set_title(image_name)

    axarr[0][i].axis('off')

    

# print test images in row 2    

test_image_names = os.listdir(test_images_path)

for i in range(3):

    im_read = cv2.imread(test_images_path + test_image_names[i],1)

    im = cv2.cvtColor(im_read, cv2.COLOR_BGR2RGB)

    axarr[1][i].imshow(im)

    axarr[1][i].set_title(image_name)

    axarr[1][i].axis('off')

plt.show()
car_models_json_names = os.listdir(car_models_json_path)

with open(car_models_json_path + car_model_json_names[1]) as json_file:

    jsonData = json.load(json_file)
vertices = np.asarray(jsonData['vertices'])

faces = np.array(jsonData['faces']) - 1

plt.figure()

ax = plt.axes(projection='3d')

ax.set_title(jsonData['car_type'])

ax.plot_trisurf(vertices[:,0], vertices[:,2], faces, -vertices[:,1])

ax.axis('off')

plt.show()