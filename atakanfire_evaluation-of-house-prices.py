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
train.head()
test.head()
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing.head(20)
tolerance = 0.8

missi = [] # Missings index



for i, per in enumerate(percent):    

    if per > tolerance:

        # print(i, per)

        missi.append(i)

        

print("Ignored Columns")

print( missing.index[missi])
train = pd.read_csv("../input/train.csv") # refresh train data

test = pd.read_csv("../input/test.csv") # refresh test data
train.drop(missing.index[missi], axis=1, inplace=True)

train.head()
for tcol in train:

    # print(tcol)

    train[tcol].fillna(0, inplace=True)
train.head()
total = test.isnull().sum().sort_values(ascending=False)

percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing.head(20)
tolerance = 0.8

missi = [] # Missings index



for i, per in enumerate(percent):    

    if per > tolerance:

        # print(i, per)

        missi.append(i)

        

print("Ignored Columns")

print( missing.index[missi])
test.drop(missing.index[missi], axis=1, inplace=True)

test.head()
for tcol in test:

    # print(tcol)

    test[tcol].fillna(0, inplace=True)

    

test = test.dropna(axis=1)

test.head()
correlations = train.corr()

correlations = correlations["SalePrice"].sort_values(ascending=False)

correlations
tolerance = 0.5

features = [] 



for i, c in enumerate(correlations):    

    if c > tolerance:

        print(i, c)

        features.append(correlations.index[i])



print(features)
import seaborn as sns

from matplotlib import pyplot as plt
plt.figure(figsize=(12,8))

sns.distplot(train['SalePrice'], color='cyan')

plt.title('Distribution of Sales Price', fontsize=18)

plt.show()
f,ax= plt.subplots(figsize=(5,5))

sns.heatmap(train.loc[:,features].corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()
data = pd.concat([train['SalePrice'], train[features[1]]], axis=1)

data.plot.scatter(x=features[1], y='SalePrice', ylim=(0,800000));
from sklearn.ensemble import RandomForestRegressor
trainy = train.SalePrice

predictors = [features[1],features[2], features[3], features[4]]

trainx = train[predictors]



model = RandomForestRegressor()

model.fit(trainx, trainy)



testx = test[predictors]

# print(testx)



predictedprices = model.predict(testx)

print("Predicted Prices")

print(predictedprices)
submission = pd.DataFrame({'ID': test.Id, 'SalePrice': predictedprices})

print(submission)



#submission.to_csv('Evaluation of House Prices: Submission.csv',index=False)
train
test