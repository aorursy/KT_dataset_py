# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# basically libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# image libaries

import cv2

import matplotlib.pyplot as plt



# for split train and test

from sklearn.model_selection import train_test_split



# for model

import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten

from tensorflow.keras.layers import Add, Concatenate, GlobalAvgPool2D

from tensorflow.keras.layers import MaxPooling2D, SeparableConv2D 

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Label.csv

labels_ds = pd.read_csv(filepath_or_buffer='../input/labels.csv', dtype={'attribute_id':np.object, 'attribute_name':np.object})

print(labels_ds.head())

print(labels_ds.tail())

print("")

print(labels_ds.info())

# train.csv

train_ds = pd.read_csv(filepath_or_buffer='../input/train.csv')

print(train_ds.head())

print("")

print(train_ds.info())

print("")

print(train_ds.head())
print(os.listdir("../input/train/")[0:12])

# image data 

# Check image data size and image by first 12 files

image_file_list = os.listdir("../input/train/")[0:12]

image_data_list = []



fig = plt.figure(figsize=(10, 15))

for image_index in range(12):

    image_file_name = train_ds.iloc[image_index, 0]

    image_np = cv2.imread("../input/train/" + image_file_name + ".png")



    image_label = "{}\n height:{} width:{}\nattr:{}".format(

        image_file_name, image_np.shape[0], image_np.shape[1], train_ds.iloc[image_index, 1]

    )

    image_area = fig.add_subplot(4,3,image_index + 1, title=image_label)

    image_area.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    

fig.tight_layout()

fig.show()
# One hot encoding for multi labels.

def OneHotEncoding(rec):

    attribute_id_list = rec["attribute_ids"].split()

    for attribute_id in attribute_id_list:

        rec[attribute_id] = 1

    

    return rec



# Append new columns from list

def AppendColumns(df, columnList):

    for newColumn in columnList:

        df[newColumn] = 0

    

    return df
# Create MultiLabelBinarizer instance and fit to attibute id in labels.csv

train_ds_encoded = AppendColumns(train_ds, labels_ds['attribute_id'])

train_ds_encoded = train_ds_encoded.apply(OneHotEncoding, axis=1)

train_ds_encoded.head()
# Append filename column

train_ds_encoded["filename"] = train_ds_encoded["id"] + ".png" 

train_ds_encoded.head()
summary_df = pd.DataFrame(data={'id':labels_ds['attribute_id'], 'attibute':labels_ds['attribute_name'], 'count':np.array(train_ds_encoded.iloc[:, 2:].sum(numeric_only=True))})

summary_df = summary_df.sort_values(by='count')

print(summary_df.head())

print(summary_df.tail(20))

rare_attr_df = summary_df.sort_values(by='count').loc[summary_df['count'] <= 5]

rare_data_df = train_ds_encoded.loc[train_ds_encoded.apply(lambda x: set(x['attribute_ids'].split(' ')).isdisjoint(rare_attr_df['id']) == False, axis=1)]

rare_data_df
train_df_2 = train_ds_encoded

for count in range(10):

    train_df_2 = train_df_2.append(rare_data_df)



train_df_2.info()
# Separate data and label

train_df_X = train_df_2.iloc[:, 0]

train_df_y = train_df_2.iloc[:, 2:]



# Split train and test

X_train, X_test, y_train, y_test = train_test_split(train_df_X, train_df_y, test_size=0.10, random_state=42)
train_df_train = train_df_2.sample(frac=0.9, random_state=42)

train_df_test = train_df_2.drop(train_df_train.index)

print("{} {} {}".format(len(train_df_2), len(train_df_train), len(train_df_test)))
# Split lable

splitted_attr = labels_ds['attribute_name'].str.split('::', expand = True)

splitted_attr.columns = ['main', 'sub']

splitted_attr
print(splitted_attr['main'].drop_duplicates())

print('culture : {}; tag : {}'.format(len(splitted_attr.loc[splitted_attr.main == 'culture']), len(splitted_attr.loc[splitted_attr.main == 'tag'])))
print(splitted_attr['sub'].drop_duplicates())

train_ds_encoded.head()
corr_df = train_ds_encoded.iloc[:, 2:-1].corr()

corr_df.head()
corr_df2 = corr_df.replace(1, 0).abs()

corr_df2['id'] = corr_df2.index

corr_df2.head()
corr_df3 = corr_df2.loc[lambda x: x[0:-1].max() > 0.4]

max_values = corr_df3.iloc[:, 0:-1].max(axis=1)

max_index1 = corr_df3.iloc[:, 0:-1].idxmax(axis=1)

max_index2 = corr_df3['id']

corr_df4 = pd.DataFrame(np.stack((max_values, max_index1, max_index2), axis=-1), columns=['value', 'id1', 'id2'])

corr_df4 = corr_df4.merge(labels_ds, left_on = 'id1', right_on = 'attribute_id')

corr_df4 = corr_df4.merge(labels_ds, left_on = 'id2', right_on = 'attribute_id', suffixes=('_1', '_2'))

corr_df4 = corr_df4.drop(columns=['attribute_id_1', 'attribute_id_2'])

corr_df4