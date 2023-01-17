# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

train_data.head()
sns.countplot(train_data.category_id)
train_data['category_id'].value_counts()
empty_img = train_data[train_data.category_id == 0]
print(empty_img.shape)

empty_img.head()
reduced_empty_img = empty_img.sample(15000)

print(reduced_empty_img.shape)
non_empty_img = train_data[train_data.category_id !=0]

print(non_empty_img.shape)
reduced_data = pd.concat((reduced_empty_img,non_empty_img))

print(reduced_data.shape)

reduced_data.head()
sns.countplot(reduced_data.category_id)
test_data = pd.read_csv('../input/test.csv')

test_data.head()
train_files = reduced_data['file_name'].values

test_files = test_data['file_name'].values
print(train_files.shape)

print(test_files.shape)
train_files[0]
train_images = list()

for i in range(len(train_files)):

    if i % 1000 == 0:

        print(i)

    img = plt.imread(os.path.join('../input/train_images/', train_files[i]))

    if len(img.shape) != 3:

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    train_images.append(cv2.resize(img,(32,32)))
test_images = list()

for i in range(len(test_files)):

    if i % 1000 == 0:

        print(i)

    img = plt.imread(os.path.join('../input/test_images/', test_files[i]))

    if len(img.shape) != 3:

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    test_images.append(cv2.resize(img,(32,32)))
from keras.utils.np_utils import to_categorical



X_train = np.array(train_images)

X_test = np.array(test_images)



target_dummies = reduced_data['category_id'].values

Y_train = to_categorical(target_dummies)



print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)
np.save('X_train.npy', X_train)

np.save('X_test.npy', X_test)

np.save('Y_train.npy', Y_train)