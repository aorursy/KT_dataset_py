import numpy as np

import pandas as pd

import cv2

from matplotlib import pyplot as plt

import random

import os

print(os.listdir('../input/dog-cat-recognition'))



input_dir = '../input/dog-cat-recognition/'
# read train labels.

train_df = pd.read_csv(input_dir + 'train_label.csv')



# make each label dataset.

dog_df = train_df[train_df.label==1].reset_index(inplace=False, drop=True)

cat_df = train_df[train_df.label==0].reset_index(inplace=False, drop=True)



display(dog_df.head())

display(cat_df.head())



print('dog size:', len(dog_df))

print('cat size:', len(cat_df))
def _prep(filename):

    # prep img.

    img = cv2.imread(input_dir + 'train/' + filename, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    _height, _width, _channel = img.shape

    title = '{}: {}x{}'.format(filename, _height, _width)

    return img, title



def _show(datas):

    # show img.

    plt.imshow(datas[0]) # img

    plt.title(datas[1]) # title

    

def show_img(data, size):

    plt.figure(figsize=(16,8))

    

    indexs = random.sample(range(len(data)), k=size)

    

    for i, index in enumerate(indexs):

        plt.subplot(1, size, i+1)

        filename = data.iloc[index]['filename']

        _show(_prep(filename))

    plt.show()
show_img(dog_df, size=5)

show_img(cat_df, size=5)