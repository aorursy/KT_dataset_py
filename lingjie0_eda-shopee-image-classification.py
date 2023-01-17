import numpy as np

import pandas as pd

import cv2

from keras.preprocessing import image

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
#settings

pd.options.display.max_columns = 999
train = pd.read_csv('../input/shopee-product-detection-student/train.csv')

test = pd.read_csv('../input/shopee-product-detection-student/test.csv')
train['category'] = train['category'].astype('object')

category_replace = {0:'00',1:'01',2:'02',3:'03',4:'04',5:'05',6:'06',7:'07',8:'08',9:'09'}

train['category'] = train['category'].replace(category_replace.keys(), category_replace.values()).astype('str')
train.info()
train.head()
fig, [ax1, ax2] = plt.subplots(1,2, figsize=(10,4))



train['category'].astype('int').plot.hist(bins=42, ax=ax1);

plt.xlabel('category');

print(train['category'].describe())

print('\nmedian counts: {:.2f}'.format(train['category'].value_counts().median()))

print('mean counts: {:.2f}'.format(train['category'].value_counts().mean()))

ax2.hist(train['category'].value_counts(), bins=42)

ax1.set_xlabel('category'), ax2.set_xlabel('counts')



pd.DataFrame(train['category'].value_counts().sort_values()).T
PATH = "../input/shopee-product-detection-student/train/train/train/"
#we'll plot 10 random photos everytime we run this command

fig, axes = plt.subplots(1,10,figsize=(15,5))

for [filename, category], ax in zip(train.values[np.random.randint(low=1,high=train.shape[0],size=10)], axes):

    img = image.load_img(PATH+category+'/'+filename, target_size=(300,300))

    ax.imshow(img), ax.text(0,0,category,c='red'), ax.axis('off')
#we'll plot 10 random photos everytime we run this command

fig, axes = plt.subplots(1,10,figsize=(15,5))

for [filename, category], ax in zip(train[train['category'] == '33'].values[np.random.randint(low=1,high=train[train['category'] == '33'].shape[0],size=10)], axes):

    img = image.load_img(PATH+category+'/'+filename, target_size=(300,300))

    ax.imshow(img), ax.text(0,0,category,c='red'), ax.axis('off')
#we'll plot 10 random photos everytime we run this command

fig, axes = plt.subplots(1,10,figsize=(15,5))

for [filename, category], ax in zip(train[train['category'] == '17'].values[np.random.randint(low=1,high=train[train['category'] == '33'].shape[0],size=10)], axes):

    img = image.load_img(PATH+category+'/'+filename, target_size=(300,300))

    ax.imshow(img), ax.text(0,0,category,c='red'), ax.axis('off')
#we'll plot 10 random photos everytime we run this command

fig, axes = plt.subplots(1,10,figsize=(15,5))

for [filename, category], ax in zip(train[train['category'] == '37'].values[np.random.randint(low=1,high=train[train['category'] == '33'].shape[0],size=10)], axes):

    img = image.load_img(PATH+category+'/'+filename, target_size=(300,300))

    ax.imshow(img), ax.text(0,0,category,c='red'), ax.axis('off')
#we'll plot 10 random photos everytime we run this command

fig, axes = plt.subplots(1,10,figsize=(15,5))

for [filename, category], ax in zip(train[train['category'] == '30'].values[np.random.randint(low=1,high=train[train['category'] == '33'].shape[0],size=10)], axes):

    img = image.load_img(PATH+category+'/'+filename, target_size=(300,300))

    ax.imshow(img), ax.text(0,0,category,c='red'), ax.axis('off')
#we'll plot 10 random photos everytime we run this command

fig, axes = plt.subplots(1,10,figsize=(15,5))

for [filename, category], ax in zip(train[train['category'] == '24'].values[np.random.randint(low=1,high=train[train['category'] == '33'].shape[0],size=10)], axes):

    img = image.load_img(PATH+category+'/'+filename, target_size=(300,300))

    ax.imshow(img), ax.text(0,0,category,c='red'), ax.axis('off')
#we'll plot 10 random photos everytime we run this command

fig, axes = plt.subplots(1,10,figsize=(15,5))

for [filename, category], ax in zip(train[train['category'] == '03'].values[np.random.randint(low=1,high=train[train['category'] == '33'].shape[0],size=10)], axes):

    img = image.load_img(PATH+category+'/'+filename, target_size=(300,300))

    ax.imshow(img), ax.text(0,0,category,c='red'), ax.axis('off')