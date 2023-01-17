# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Matplotlib Pyplot

import seaborn as sns # Plotting seaborn

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

train.head()
train.index = train['id']

train = train.drop(columns='id')

train.head()
train.isna().sum()
sns.pairplot(train)
train.describe()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error



def rmse(y_true, y_pred):

    return mean_squared_error(y_true,y_pred)**0.5
plt.figure(figsize=(12,10))

sns.heatmap(train.corr(), annot=True, fmt='.03f',linewidths=.5)
x = train.drop(columns=['quality','citric acid','free sulfur dioxide','sulphates'])

y = train['quality']
x_scaled = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.3, random_state=666)
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x_train, y_train)

pred = reg.predict(x_test)

rmse(y_test, pred)
#load test dan skalakan

test = pd.read_csv('../input/test.csv')

test.index = test['id']

test = test.drop(columns=['id','citric acid','free sulfur dioxide','sulphates'])

test.head()
test_scaled = StandardScaler().fit_transform(test)

ans = reg.predict(test_scaled)
sub = pd.read_csv('../input/sample_submission.csv')

sub.index = sub['id']

sub = sub.drop(columns='id')

sub['quality'] = ans

sub.head()
sub.to_csv('submission.csv')