# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.image as mpimg

from matplotlib import pyplot as plt

import seaborn as sns

import albumentations as A



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

test_df = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

sample_sub_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
train_df = train_df.drop(['grapheme'], axis=1, inplace=False)

train_df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] = train_df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].astype('uint8')
def imageProcessing(df, size=256):

    imageProcessed = {}

    

    for i in (range(df.shape[0])):

        #Interpolation

        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size),interpolation=cv2.INTER_AREA)

        #Noise Removing

        image=cv2.fastNlMeansDenoising(image)

        #Gaussian Blur

        gaussian_3 = cv2.GaussianBlur(image, (9,9), 10.0) #unblur

        image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)

        #Laplacian Filter

        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) #filter

        image = cv2.filter2D(image, -1, kernel)

        #Otsu Method for Thresholding

        ret,image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        imageProcessed[df.index[i]] = image.reshape(-1)

   

    imageProcessed = pd.DataFrame(imageProcessed).T

    return imageProcessed
train_images = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_0.parquet'), train_df, on='image_id').drop(['image_id'], axis=1)



# Visualize few samples of current training dataset

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(20, 10))

count=0

for row in ax:

    for col in row:

        col.imshow(imageProcessing(train_images.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).iloc[[count]]).values.reshape(256, 256))

        count += 1

plt.show()
train_images = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_1.parquet'), train_df, on='image_id').drop(['image_id'], axis=1)



# Visualize few samples of current training dataset

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20, 10))

count=0

for row in ax:

    for col in row:

        col.imshow(imageProcessing(train_images.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).iloc[[count]]).values.reshape(256, 256))

        count += 1

plt.show()
train_images = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_2.parquet'), train_df, on='image_id').drop(['image_id'], axis=1)



# Visualize few samples of current training dataset

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20, 10))

count=0

for row in ax:

    for col in row:

        col.imshow(imageProcessing(train_images.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).iloc[[count]]).values.reshape(256, 256))

        count += 1

plt.show()
train_images = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_3.parquet'), train_df, on='image_id').drop(['image_id'], axis=1)



# Visualize few samples of current training dataset

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20, 10))

count=0

for row in ax:

    for col in row:

        col.imshow(imageProcessing(train_images.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).iloc[[count]]).values.reshape(256, 256))

        count += 1

plt.show()
def crop_images(df, size=64):

    resized = {}

    resize_size=64

    

    for i in range(df.shape[0]):

        #image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size),None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)

        image=df.loc[df.index[i]].values.reshape(137,236)

        #Removing Blur

        aug = A.GaussianBlur(p=1.0)

        image = aug(image=image)['image']

        #Noise Removing

        #augNoise=A.MultiplicativeNoise(p=1.0)

        #image = augNoise(image=image)['image']

        #Removing Distortion

        #augDist=A.ElasticTransform(sigma=50, alpha=1, alpha_affine=10, p=1.0)

        #image = augDist(image=image)['image']

        #Brightness

        augBright=A.RandomBrightnessContrast(p=1.0)

        image = augBright(image=image)['image']

        #Thresholding

        ret, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]



        idx = 0 

        ls_xmin = []

        ls_ymin = []

        ls_xmax = []

        ls_ymax = []

        for cnt in contours:

            idx += 1

            x,y,w,h = cv2.boundingRect(cnt)

            ls_xmin.append(x)

            ls_ymin.append(y)

            ls_xmax.append(x + w)

            ls_ymax.append(y + h)

        xmin = min(ls_xmin)

        ymin = min(ls_ymin)

        xmax = max(ls_xmax)

        ymax = max(ls_ymax)



        roi = image[ymin:ymax,xmin:xmax]

        resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)

        #image=affine_image(image)

        #image= crop_resize(image)

        #image = cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA)

        #image=resize_image(image,(64,64))

        #image = cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA)

        #gaussian_3 = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT) #unblur

        #image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)

        #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) #filter

        #image = cv2.filter2D(image, -1, kernel)

        #ret,image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        resized[df.index[i]] = resized_roi.reshape(-1)

    resized = pd.DataFrame(resized).T

    return resized
train_images = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_0.parquet'), train_df, on='image_id').drop(['image_id'], axis=1)



# Visualize few samples of current training dataset

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(20, 10))

count=0

for row in ax:

    for col in row:

        col.imshow(crop_images(train_images.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).iloc[[count]]).values.reshape(64, 64))

        count += 1

plt.show()
train_images = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_1.parquet'), train_df, on='image_id').drop(['image_id'], axis=1)



# Visualize few samples of current training dataset

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(20, 10))

count=0

for row in ax:

    for col in row:

        col.imshow(crop_images(train_images.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).iloc[[count]]).values.reshape(64, 64))

        count += 1

plt.show()
train_images = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_2.parquet'), train_df, on='image_id').drop(['image_id'], axis=1)



# Visualize few samples of current training dataset

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(20, 10))

count=0

for row in ax:

    for col in row:

        col.imshow(crop_images(train_images.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).iloc[[count]]).values.reshape(64, 64))

        count += 1

plt.show()
train_images = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_3.parquet'), train_df, on='image_id').drop(['image_id'], axis=1)



# Visualize few samples of current training dataset

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(20, 10))

count=0

for row in ax:

    for col in row:

        col.imshow(crop_images(train_images.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).iloc[[count]]).values.reshape(64, 64))

        count += 1

plt.show()