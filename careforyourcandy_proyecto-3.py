# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.isnull().sum()

print(train.shape)

print(test.shape)

print("Valores null en train set")

print(train.isnull().sum())

print("Valores null en test set")

print(test.isnull().sum())
train, dev = train_test_split(train, test_size=0.1) 



#configuration values 





#Extract X. Values transforms it into array

X_train = train['question_text'].values

X_dev = dev['question_text'].values

x_test = test['question_text'].values

#Extract Y

Y_train = train['target'].values

Y_dev = dev['target'].values