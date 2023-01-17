# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# no col titles, in normal task that  mast be provide 

train = pd.read_csv("../input/adult-training.csv", header=None)
train.sample(10)
def transform_data_1(df):

    # turn all in numerical

    cat_columns = [1, 3, 5, 6, 7, 8, 9, 13, 14]

    df[cat_columns] = df[cat_columns].astype('category')

    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    scaler = MinMaxScaler()

    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df
train = transform_data_1(train)
# corelation matrix

corr = train.corr()

corr.style.background_gradient(cmap='coolwarm')
# i will take cols 0, 4, 5, 7, 9, 10, 11, 12

def transform_data_2(df):

    # drop res of colums

    df = df.drop([1,2,3,6,8,13], axis=1)

    return df
train = transform_data_2(train)

train.sample(5)
# chcek for nan values

train.isnull().any().any()
# spliting data

y_train_all = train[14]

train = train.drop(14, axis=1)

x_train, x_test, y_train, y_test = train_test_split(train, y_train_all, test_size=0.2, random_state=42)
y_test.shape
# logistic regresion sklearn model

model = LogisticRegression(max_iter = 100, multi_class='ovr', solver='lbfgs')
model.fit(x_train, y_train)
model.score(x_test, y_test)