from IPython.display import Image

Image("../input/imagefile/place.jpg")
#Importing necessary libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

import glob
import os

print(os.listdir("../input/landmark-retrieval-2020"))
#reading the training and test data

train_data = pd.read_csv('../input/landmark-retrieval-2020/train.csv')



print("Training data size:",train_data.shape)
train_data.head()
train_data.info()
train_data.head()

train_data['landmark_id'][33]
#Displaying number of unique URLs & ids

len(train_data['landmark_id'].unique())
len(train_data['id'].unique())
plt.title('Distribution')

sns.distplot(train_data['landmark_id'])
sns.set()

print(train_data.nunique())

train_data['landmark_id'].value_counts().hist()
from scipy import stats

sns.set()

res = stats.probplot(train_data['landmark_id'], plot=plt)
test_list = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')

index_list = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')

train_list= glob.glob('../input/landmark-retrieval-2020/train/*/*/*/*')
plt.rcParams["axes.grid"] = True

f, axarr = plt.subplots(6, 5, figsize=(24, 22))



curr_row = 0

for i in range(30):

    example = cv2.imread(test_list[i])

    example = example[:,:,::-1]

    

    col = i%6

    axarr[col, curr_row].imshow(example)

    if col == 5:

        curr_row += 1