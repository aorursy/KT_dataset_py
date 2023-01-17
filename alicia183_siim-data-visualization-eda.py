import numpy as np

import pandas as pd

import os

from os import listdir

from os.path import isfile, join

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

import pydicom

from sklearn.impute import SimpleImputer

print("Complete")
train_jpeg_dir = '../input/siim-isic-melanoma-classification/jpeg/train/'

train_jpeg = [f for f in listdir(train_jpeg_dir) if isfile(join(train_jpeg_dir, f))]



test_jpeg_dir = '../input/siim-isic-melanoma-classification/jpeg/test/'

test_jpeg = [f for f in listdir(test_jpeg_dir) if isfile(join(test_jpeg_dir, f))]
train_dcm_dir = '../input/siim-isic-melanoma-classification/train/'

train_dcm = [f for f in listdir(train_dcm_dir) if isfile(join(train_dcm_dir, f))]



test_dcm_dir = '../input/siim-isic-melanoma-classification/test/'

test_dcm = [f for f in listdir(test_dcm_dir) if isfile(join(test_dcm_dir, f))]
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
train.head()
test.head()
fig=plt.figure(figsize=(15, 10))

columns = 4

rows = 3

for i in range(1, columns*rows +1):

    path = train_jpeg_dir + train_jpeg[i]

    fig.add_subplot(rows, columns, i)

    plt.imshow(mpimg.imread(path))

    fig.add_subplot
fig=plt.figure(figsize=(15, 10))

columns = 4

rows = 3

for i in range(1, columns*rows +1):

    ds = pydicom.dcmread(train_dcm_dir + train_dcm[i])

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)

    fig.add_subplot
fig=plt.figure(figsize=(15, 10))

columns = 4

rows = 3

for i in range(1, columns*rows +1):

    path = test_jpeg_dir + test_jpeg[i]

    fig.add_subplot(rows, columns, i)

    plt.imshow(mpimg.imread(path))

    fig.add_subplot
fig=plt.figure(figsize=(15, 10))

columns = 4

rows = 3

for i in range(1, columns*rows +1):

    ds = pydicom.dcmread(test_dcm_dir + test_dcm[i])

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)

    fig.add_subplot
features_first = ["sex", "age_approx", "anatom_site_general_challenge"]

features_train = ["diagnosis", "benign_malignant", "target"]

features = ["sex", "age_approx", "anatom_site_general_challenge", "diagnosis", "benign_malignant", "target"]



sns.set(style="ticks", color_codes=True)

fig = plt.gcf()

fig.set_size_inches(15, 10)
for i in features_first:

    sns.set(font_scale=0.6)

    plt.title("Count of " + i + " train")

    sns.catplot(x = i, kind="count", palette="ch:.25", data=train)

    

    sns.set(font_scale=0.6)

    plt.title("Count of " + i + " test")

    sns.catplot(x = i, kind="count", palette="ch:.25", data=test)
for i in features_train:

    sns.set(font_scale=0.6)

    plt.title("Count of " + i + " train")

    sns.catplot(x = i, kind="count", palette="ch:.25", data=train)
for i in features_first:

    sns.set(font_scale=0.7)

    plt.title("belign_malignant for " + i)

    sns.catplot(x=i,kind='count', hue = "benign_malignant", palette="ch:.25", data=train)
for i in features_first:

    sns.set(font_scale=0.7)

    plt.title("target for " + i)

    sns.catplot(x=i,kind='count', hue = "target", palette="ch:.25", data = train)
for i in features_first:

    sns.set(font_scale=0.7)

    plt.title("melanoma for " + i)

    sns.catplot(x=i,kind= 'count', hue= "diagnosis", palette="ch:.25", data = train)
print('Train Set')

print(train.info())
print('Test Set')

print(test.info())
imp_mean_train = SimpleImputer( strategy='most_frequent')

train_no_null = pd.DataFrame(imp_mean_train.fit_transform(train))

train_no_null.columns=train.columns

train_no_null.index=train.index

train_no_null.head()
imp_mean_test = SimpleImputer( strategy='most_frequent')

test_no_null = pd.DataFrame(imp_mean_train.fit_transform(test))

test_no_null.columns=test.columns

test_no_null.index=test.index

test_no_null.head()
for i in features:

    sns.set(font_scale=0.6)

    plt.title("Count of " + i + " without filling missing values")

    sns.catplot(x = i, kind="count", palette="ch:.25", data=train)

    

    sns.set(font_scale=0.6)

    plt.title("Count of " + i + " filling missing values")

    sns.catplot(x = i, kind="count", palette="ch:.25", data=train_no_null)
for i in features_first:

    sns.set(font_scale=0.6)

    plt.title("Count of " + i + " without filling missing values")

    sns.catplot(x = i, kind="count", palette="ch:.25", data=test)

    

    sns.set(font_scale=0.6)

    plt.title("Count of " + i + " filling missing values")

    sns.catplot(x = i, kind="count", palette="ch:.25", data=test_no_null)