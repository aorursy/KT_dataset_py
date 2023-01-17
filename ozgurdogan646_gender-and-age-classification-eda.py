# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
#load data

data = pd.read_csv("/kaggle/input/adience-benchmark-gender-and-age-classification/AdienceBenchmarkGenderAndAgeClassification/fold_3_data.txt",sep = "\t" )

data1 = pd.read_csv("/kaggle/input/adience-benchmark-gender-and-age-classification/AdienceBenchmarkGenderAndAgeClassification/fold_1_data.txt",sep = "\t")

data2 = pd.read_csv("/kaggle/input/adience-benchmark-gender-and-age-classification/AdienceBenchmarkGenderAndAgeClassification/fold_2_data.txt",sep = "\t")

data3 = pd.read_csv("/kaggle/input/adience-benchmark-gender-and-age-classification/AdienceBenchmarkGenderAndAgeClassification/fold_0_data.txt",sep = "\t")

data4 = pd.read_csv("/kaggle/input/adience-benchmark-gender-and-age-classification/AdienceBenchmarkGenderAndAgeClassification/fold_4_data.txt",sep = "\t")
data.shape

total_shape = (data.shape[0]+data1.shape[0]+data2.shape[0]+data3.shape[0]+data4.shape[0],data.shape[1])

print(data4.shape)

print(total_shape)
data.columns
data.dtypes
data.describe()
#first 10 rows of data

data.head(10)
#pie_graph

plt.figure(1, figsize=(8,8))

data.age.value_counts().plot.pie(autopct="%1.1f%%")

plt.show()
#bar chart

gender = ['f','m','u']

plt.bar(gender,data.gender.value_counts(), align='center', alpha=0.5)

plt.show()
path = "/kaggle/input/adience-benchmark-gender-and-age-classification/AdienceBenchmarkGenderAndAgeClassification/faces/101071073@N04/coarse_tilt_aligned_face.708.10656436223_37c5dafe60_o.jpg"

img = load_img(path)

plt.imshow(img)

plt.show()
path2 = "/kaggle/input/adience-benchmark-gender-and-age-classification/AdienceBenchmarkGenderAndAgeClassification/faces/101071073@N04/coarse_tilt_aligned_face.708.10658353233_02f5201237_o.jpg"

img = load_img(path2)

plt.imshow(img)

plt.show()