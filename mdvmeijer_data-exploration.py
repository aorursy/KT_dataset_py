# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import *

import random, math

import multiprocessing

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



xtrain = pd.read_csv('../input/X_train.csv')

ytrain = pd.read_csv('../input/y_train.csv')

train = pd.merge(xtrain, ytrain, how='left', on='series_id')



xtest = pd.read_csv('../input/X_test.csv')

ytest = pd.read_csv('../input/sample_submission.csv')

test = pd.merge(xtest, ytest, how='left', on='series_id')

print(train.shape, test.shape)



# Any results you write to the current directory are saved as output.
train.head()
test.head()
sns.set(style='darkgrid')

sns.countplot(y = 'surface',

              data = train,

              order = train['surface'].value_counts().index)

plt.show()
seriesFineConcrete = train[train["series_id"]==0]

seriesConcrete = train[train["series_id"]==1]

seriesSoftTiles = train[train["series_id"]==4]
seriesFineConcrete.shape
seriesFineConcrete.head()
plt.figure(figsize=(26, 16))

for i, col in enumerate(seriesFineConcrete.columns[3:-2]):

    plt.subplot(3, 4, i + 1)

    plt.plot(seriesFineConcrete[col], 'r')

    plt.plot(seriesConcrete[col], 'g')

    plt.plot(seriesSoftTiles[col], 'b')

    plt.title(col)