import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split #Cross validation





%config InlineBackend.figure_format = 'png' #set 'png' here when working on notebook

%matplotlib inline



train = pd.read_csv("../input/adult.csv")
train.head()
train.describe()
train['age'].hist(bins = 100)
capitals = pd.DataFrame({"gains":train['capital.gain'], "losses":train['capital.loss']})

capitals.hist(bins = 12)
train['income'] = train['income'].replace({'<=50K': 0, '>50K':1}, regex=True)
train.head()
train = pd.get_dummies(train)
print(train.shape)
train.head()
data = train.drop('income')

target = train.income

#X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state=0)