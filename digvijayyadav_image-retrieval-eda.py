# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential

import glob

import seaborn as sns

import plotly.graph_objects as go

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data_df = pd.read_csv('../input/landmark-retrieval-2020/train.csv')

print('The Size of csv file : {}'.format(train_data_df.shape))
train_data_df.head(5)
print('The missing values in dataset : {}'.format(train_data_df.isnull().sum().sum()))

print('The Nan values in dataset : {}'.format(train_data_df.isna().sum().sum()))
describe = train_data_df.describe()
plt.figure(figsize=(10,10))

describe.plot(kind='bar')
train_data_df_count = pd.DataFrame(train_data_df.landmark_id.value_counts().sort_values(ascending=False))

train_data_df_count.reset_index(inplace=True)

train_data_df_count.columns = ['landmark_id', 'count']

train_data_df_count
sns.set()

plt.figure(figsize=(10,5))

sns.barplot(

    x="landmark_id",

    y="count",

    data=train_data_df_count.head(10),

    label="Count",

    order=train_data_df_count.head(10).landmark_id)

locs, labels = plt.xticks()

plt.setp(labels, rotation=45)

plt.show()
#to check the distribution of data we use kurtosis

train_data_df.kurtosis()
from cv2 import cv2

im = cv2.imread('../input/landmark-retrieval-2020/train/0/0/0/0000059611c7d079.jpg')

plt.imshow(im)
# From: https://www.kaggle.com/sudeepshouche/identify-landmark-name-from-landmark-id

url = 'https://s3.amazonaws.com/google-landmark/metadata/train_label_to_category.csv'

df_class = pd.read_csv(url, index_col = 'landmark_id', encoding='latin', engine='python')['category'].to_dict()
import math

def print_img(class_id, df_class, figsize):

    file_path = "../input/landmark-retrieval-2020/train/"

    df = train_data_df[train_data_df['landmark_id'] == class_id].reset_index()

    

    print("Class {} - {}".format(class_id, df_class[class_id].split(':')[-1]))

    print("Number of images: {}".format(len(df)))

    

    plt.rcParams["axes.grid"] = False

    no_row = math.ceil(min(len(df), 12)/3) 

    f, axarr = plt.subplots(no_row, 3, figsize=figsize)



    curr_row = 0

    len_img = min(12, len(df))

    for i in range(len_img):

        img_name = df['id'][i] + ".jpg"

        img_path = os.path.join(

            file_path, img_name[0], img_name[1], img_name[2], img_name)

        example = cv2.imread(img_path)

        # uncomment the following if u wanna rotate the image

        # example = cv2.rotate(example, cv2.ROTATE_180)

        example = example[:,:,::-1]



        col = i % 3

        axarr[curr_row, col].imshow(example)

        axarr[curr_row, col].set_title("{}. {} ({})".format(

            class_id, df_class[class_id].split(':')[-1], df['id'][i]))

        if col == 2:

            curr_row += 1
class_id = 138982

print_img(class_id, df_class, figsize=(25,25))
class_id =  126637

print_img(class_id, df_class, figsize=(25,25))