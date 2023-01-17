import pandas as pd

import numpy as np

import glob

import cv2

import matplotlib.pyplot as plt

import PIL

from PIL import Image, ImageDraw

import seaborn as sns

import random

import os

#from tensorflow.keras.applications import EfficientNetB0

#model = EfficientNetB0(weights='imagenet')



plt.style.use('seaborn-darkgrid')
df_train = pd.read_csv('../input/landmark-recognition-2020/train.csv')
#df_train.landmark_id.value_counts(normalize=True).plot(kind='bar')

print('Total number of class: {}'.format(df_train.landmark_id.nunique()))
#train and test directory

train_dir = "../input/landmark-recognition-2020/train/*/*/*/*.jpg"

test_dir = "../input/landmark-recognition-2020/test/*/*/*/*.jpg"
df_train.landmark_id.value_counts(normalize=False)[:10].plot(kind='bar', title='Frequency of occurrence of top 10 Labels')
train_list = glob.glob('../input/landmark-recognition-2020/train/*/*/*/*')



#train_list
#Random 12 images from train set



random_path = random.sample(train_list, 12)

plt.rcParams["axes.grid"] = False

f, axarr = plt.subplots(4, 3, figsize=(24, 22))



curr_row = 0

for i in range(12):

    example = cv2.imread(random_path[i])

    example = example[:,:,::-1]

    

    col = i%4

    axarr[col, curr_row].imshow(example)

    if col == 3:

        curr_row += 1

least_landmarks = df_train.landmark_id.value_counts()[-10:].index.values

least_ids = df_train[df_train.landmark_id.isin(least_landmarks)].id.unique()
least_ids
f, ax = plt.subplots(5,2, figsize=(18,22))

for i, image_id in enumerate(least_ids):

    image_path = os.path.join('../input/landmark-recognition-2020/train', f'{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg')

    image = cv2.imread(image_path)

    image = image[:,:,::-1]



    ax[i//5, i%2].imshow(image) 

    #image.close()       

    ax[i//5, i%2].axis('off')



    landmark_id = df_train[df_train.id==image_id.split('.')[0]].landmark_id.values[0]

    ax[i//5, i%2].set_title(f"ID: {image_id.split('.')[0]}\nLandmark_id: {landmark_id}", fontsize="12")



plt.show() 
images = cv2.imread(random_path[10])

print(images.shape)