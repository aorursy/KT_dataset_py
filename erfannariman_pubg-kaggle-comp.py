import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))
os.chdir("../input")
train = pd.read_csv('train_V2.csv')
test = pd.read_csv('test_V2.csv')
statinfo_train = os.stat('train_V2.csv')
statinfo_test = os.stat('test_V2.csv')

print('size of train file in mb\'s is', statinfo_train.st_size/10**6)
print('size of test file in mb\'s is',statinfo_test.st_size/10**6)
# show first 5 rows of training and test data
train.head()
test.head()
print("the shape of our training dataset format(#rows, #columns) =", train.shape)
print("the shape of our test dataset format(#rows, #columns) =", test.shape)
train.isna().sum()
train[train.winPlacePerc.isna()]
train.dropna(how="any", inplace=True)
train.shape
train.dtypes[train.dtypes == 'object']
print(f"amount of unique values in the matchType column is: {train.matchType.value_counts().count()}")
print("these are the unique values:")
train.matchType.value_counts()
'''
We will use one hot encoding on the matchType series, which will give each unique value its own column.
This means that we will get 16 new columns: squad-fpp, duo-fpp, squad, solo-fpp etc.
'''
one_hot = pd.get_dummies(train.matchType)
train.drop('matchType', axis=1, inplace=True)
train = train.join(one_hot)
train.head()
train.drop(['Id', 'groupId', 'matchId'], axis='columns', inplace=True)
train.corr()[round(train.corr(), 2).winPlacePerc > 0.5]
fig = plt.figure(1, figsize=(10,10))
plt.plot(train.walkDistance, train.winPlacePerc, 'o')
plt.show()
train.walkDistance.sort_values(ascending=False).head()
train[train.walkDistance == 25780]