# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for data visualization

import seaborn as sns

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

sns.set(style='darkgrid')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/image-classification-dataset-in-the-gala-event/image_auto_tagging/train.csv")

test_data = pd.read_csv("/kaggle/input/image-classification-dataset-in-the-gala-event/image_auto_tagging/test.csv")

train_data.head()
train_data.shape
train_data.isnull().sum().sum()
# plotting a histogram to identify the frequency of each type in class

count_classes = pd.value_counts(train_data['Class'], sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Credit Card - Fraud Class histogram (1 represent Fraud)")

plt.xlabel("Class")

plt.ylabel("Frequency")
# function to plot n images using subplots

def plot_image(images, captions=None, cmap=None ):

    f, axes = plt.subplots(1, len(images), sharey=True)

    f.set_figwidth(15)

    for ax,image in zip(axes, images):

        ax.imshow(image, cmap)