# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import time

import cv2

import glob

import matplotlib.pyplot as plt





print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_image_names = glob.glob('../input/gtsrb_challenge/GTSRB_Challenge/train/*/*.ppm')

print("Total number of training images: ", len(train_image_names))



# make train_image_names as serie object

train_image_names = pd.Series(train_image_names)
# train_df: a dataframe with 2 field: Filename, ClassId

train_df = pd.DataFrame()



# generate Filename field

train_df['Filename'] = train_image_names.map(lambda img_name: img_name.split("/")[-1])



# generate ClassId field

train_df['ClassId'] = train_image_names.map(lambda img_name: int(img_name.split("/")[-2]))



train_df.head()
plot_df = train_df.sample(9).reset_index()

plt.figure(figsize=(10, 10))



for i in range(9):

    img_name = plot_df.loc[i, 'Filename']

    label_str = "%05d"%(plot_df.loc[i, 'ClassId'])

    plt.subplot(3,3,i+1)

    plt.imshow(plt.imread(os.path.join('../input/gtsrb_challenge/GTSRB_Challenge/train/',label_str, img_name)))

    plt.title(label_str)

    plt.xticks([])

    plt.yticks([])
class_id_distribution = train_df['ClassId'].value_counts()

class_id_distribution.head(10)
plt.figure(figsize=(13,5))

plt.xticks(np.arange(43))

plt.bar(class_id_distribution.index, class_id_distribution.values)