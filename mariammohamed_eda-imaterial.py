import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import os

import json

import keras
print(os.listdir("../input"))
train_data = pd.read_csv('../input/train.csv')
train_data.shape
train_data.head()
print('Number of unique heights = {0}'.format(train_data.Height.unique().shape[0]))

print('Minimum height = {0}'.format(train_data.Height.min()))

print('Maximum height = {0}'.format(train_data.Height.max()))

print('Average height = {0}'.format(train_data.Height.mean()))
print('Number of unique widths = {0}'.format(train_data.Width.unique().shape[0]))

print('Minimum width = {0}'.format(train_data.Width.min()))

print('Maximum width = {0}'.format(train_data.Width.max()))

print('Average width = {0}'.format(train_data.Width.mean()))
plt.hist(train_data.groupby(['ImageId']).agg('mean')['Height'], bins=100)

plt.xlabel('Height of images')

plt.ylabel('Number of images')

plt.show()
plt.hist(train_data.groupby(['ImageId']).agg('mean')['Width'], bins=100)

plt.xlabel('Width of images')

plt.ylabel('Number of images')

plt.show()
plt.hist(train_data.groupby(['ImageId']).agg('mean')['Height']/train_data.groupby(['ImageId']).agg('mean')['Width'], bins=20)

plt.xlabel('Ratio (height / width)')

plt.ylabel('Number of images')

plt.show()
train_data.ImageId.unique().shape[0]
num_classes = train_data.groupby(by=['ImageId']).count()['ClassId'].tolist()
print('Min number of classes is {0} and maximum number is {1}'.format(min(num_classes), max(num_classes)))
plt.hist(num_classes, bins=74)

plt.xlabel('Number of classes for one image')

plt.ylabel('Number of images')

plt.show()
train_data.ClassId.unique().shape[0]
with open('../input/label_descriptions.json') as json_file:  

    labels_desc = json.load(json_file)
labels_desc.keys()
labels_desc['info']
categories = pd.DataFrame(labels_desc['categories'])
attributes = pd.DataFrame(labels_desc['attributes'])
categories.shape
attributes.shape
categories
attributes