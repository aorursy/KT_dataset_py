import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import linear_model



import os
# Change plot size

plt.rcParams["figure.figsize"] = [10, 10] #default [6, 4]
## Load input files

input_dir = "../input"

train_file = os.path.join(input_dir, "train.csv")

test_file = os.path.join(input_dir, "test.csv")



train = pd.read_csv(train_file)

test = pd.read_csv(test_file)
train.shape
train.head()
features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
fig, axs = plt.subplots(3,3)



row = -1

for i,f in enumerate(features):

    col = i % 3

    if col==0: row+=1



    train.groupby(f).size().plot(kind="bar", ax=axs[row,col])

plt.tight_layout()
fig, axs = plt.subplots(3,3)



row = -1

for i,f in enumerate(features):

    col = i % 3

    if col==0: row+=1



    train.groupby(f).size().plot(kind='bar', color='#d47577', ax=axs[row,col])

    if f=='Survived':

        s = train.groupby(f).size()

        s[0]=0

        s.plot(kind='bar', color='#75d4d2', ax=axs[row,col])

    else:

        train[train['Survived'] == 1].groupby(f).size().plot(kind='bar', color='#75d4d2', ax=axs[row,col])

plt.tight_layout()