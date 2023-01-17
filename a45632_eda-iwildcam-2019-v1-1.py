# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns # for making plots with seaborn

color = sns.color_palette()

from PIL import Image

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head(5)
print('size of train data',train.shape)

print('size of test data',test.shape)
plt.figure(figsize=(12,5))

plt.title("Distribution of image categories(Target Variable)")

ax = sns.distplot(train["category_id"])
plt.figure(figsize=(12,5))

plt.title("Distribution of train image locations(Train location Variable)")

ax = sns.distplot(train["location"])
plt.figure(figsize=(12,5))

plt.title("Distribution of test image locations(Test location Variable)")

ax = sns.distplot(test["location"])
sns.FacetGrid(train, hue="height", size=10).map(plt.scatter, "category_id", "location").add_legend()
train['date'] = train['date_captured'].str.split('\s+').str[0]

train['time'] = train['date_captured'].str.split('\s+').str[-1]    

train['hour'] = pd.to_numeric(train['time'].str[:2], errors='coerce')

sns.FacetGrid(train, hue="category_id", size=10).map(plt.scatter, "hour", "location").add_legend()
night = train[(train['hour'] > 19) | (train['hour'] < 7)]

day = len(train) - len(night)

labels = 'Day', 'Night'

sizes = [len(night), day]

colors = ['lightcoral', 'lightskyblue']

explode = (0.1, 0) 

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.show()
# sample night images

fig = plt.figure(figsize=(25, 60))

imgs = [np.random.choice(night.loc[night['category_id'] == i, 'file_name'], 4) for i in night.category_id.unique()]

imgs = [i for j in imgs for i in j]

labels = [[i] * 4 for i in train.category_id.unique()]

labels = [i for j in labels for i in j]

for idx, img in enumerate(imgs):

    ax = fig.add_subplot(14, 4, idx + 1, xticks=[], yticks=[])

    im = Image.open("../input/train_images/" + img)

    plt.imshow(im)

    ax.set_title(f'Label: {labels[idx]}')
day = train[(train['hour'] < 19) & (train['hour'] > 7)]

# sample night images

fig = plt.figure(figsize=(25, 60))

imgs = [np.random.choice(day.loc[day['category_id'] == i, 'file_name'], 4) for i in day.category_id.unique()]

imgs = [i for j in imgs for i in j]

labels = [[i] * 4 for i in train.category_id.unique()]

labels = [i for j in labels for i in j]

for idx, img in enumerate(imgs):

    ax = fig.add_subplot(14, 4, idx + 1, xticks=[], yticks=[])

    im = Image.open("../input/train_images/" + img)

    plt.imshow(im)

    ax.set_title(f'Label: {labels[idx]}')