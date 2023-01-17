# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
from collections import Counter

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

%matplotlib inline

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")[:3000]
filenames = train['Image']
labels = train['Id']
print(train.shape)
train.head()
images={filename: plt.imread(f'../input/train/{filename}') for filename in filenames}
#images=[ plt.imread(f'../input/train/{filename}') for filename in filenames]
def plot_images(filenames, labels, rows=4):
    # Set figure to 13 inches x 8 inches
    figure = plt.figure(figsize=(13, 8))

    cols = len(filenames) // rows +1

    for i in range(len(filenames)):
        subplot = figure.add_subplot(rows, cols, i+1)
        subplot.axis('Off')
        if labels:
            subplot.set_title(labels[i], fontsize=16)
        plt.imshow(images[filenames[i]], cmap='gray')


plot_images(list(filenames[:18]), list(labels[:18]))
num_categories = len(train['Id'].unique())
     
print(f'Number of categories: {num_categories}')
categories_count = train['Id'].value_counts()
size_buckets = Counter(categories_count.values)

plt.figure(figsize=(10, 6))

plt.bar(range(len(size_buckets)), list(size_buckets.values())[::-1], align='center')
plt.xticks(range(len(size_buckets)), list(size_buckets.keys())[::-1])
plt.title("Num of categories by available images in the training set")
plt.xlabel('# of images provided')
plt.ylabel('# of categories')

plt.show()
categories_count.head()
total = len(train['Id'])
new_whale_percentage = categories_count["new_whale"]/total
print(f'Total images in training set {total}')
print(f'Percentage of new_whale {new_whale_percentage*100}%')
top_names = categories_count.head().keys()
top1_name = top_names[1]
top2_name = top_names[2]
top1 = list(filenames[train['Id'] == top1_name])
plot_images(top1, None, rows=len(top1)/4)
top2 = list(images[train['Id'] == top2_name])
plot_images(top2, None, rows=len(top1)/4)
num_grey_scale = 0
num_grey_scale_unpopular = 0
for key, value in images.items():
    if value.ndim == 2:
        num_grey_scale += 1
        if not key in top_names:
            num_grey_scale_unpopular +=1
        
percentage_grey_scale = num_grey_scale/total
percentage_grey_scale_unpopular = num_grey_scale_unpopular/total

print(f'Percentage of grey scale images  {percentage_grey_scale*100}%')
print(f'Percentage of small data grey scale images  {percentage_grey_scale_unpopular*100}%')
img_sizes = Counter([value.shape[:2] for value in images.values()])

size, freq = zip(*Counter({i: v for i, v in img_sizes.items() if v > 1}).most_common(20))

plt.figure(figsize=(10, 6))

plt.bar(range(len(freq)), list(freq), align='center')
plt.xticks(range(len(size)), list(size), rotation=70)
plt.title("Image size frequencies (where freq > 1)")

plt.show()