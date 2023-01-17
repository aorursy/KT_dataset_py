# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print("Number of rows and columns in train set : ",train.shape)

print("Number of rows and columns in test set : ",test.shape)
train_ratio = []

test_ratio = []

feats = [f for f in train.columns if f not in ['id','target']]

for i in feats:

    train_ratio.append(train[i].nunique()/train.shape[0])

    test_ratio.append(test[i].nunique()/test.shape[0])
np.sort(train_ratio)[:10]
np.sort(test_ratio)[:10]
train.columns[np.argmin(train_ratio) + 1]
train['wheezy-copper-turtle-magic'].head(10)
train['wheezy-copper-turtle-magic'].value_counts().head(10)