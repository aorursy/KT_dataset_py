# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn as sk

from sklearn import datasets, linear_model, ensemble, dummy, neighbors, tree, feature_selection,decomposition,preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt

import pickle

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/bits-f464-l1/train.csv")

print(train_df.head())

columns_dropped = []
test_df = pd.read_csv("../input/bits-f464-l1/test.csv")

test_df.head()
test_df.fillna(test_df.mean(),inplace=True)
train_df.info()
train_df.describe()
print(train_df.isnull().values.any())

for column in train_df:

    if train_df[column].isnull().values.any() == True:

        print(column)

        print(train_df[column].unique())

        #train_df.drop(inplace=True,columns=column)

train_df.fillna(train_df.mean(),inplace=True)
for column in train_df:

    if train_df[column].unique().size == 1:

        print(column)

        columns_dropped.append(column)

        train_df.drop(inplace=True,columns=column)
test_df.drop(inplace=True,columns=columns_dropped)
print(test_df.isnull().values.any())
test_df.describe()
train_df.describe()
columns_dropped
train_df.select_dtypes(include='int64')
train_df.corr()
for column in train_df:

    if column == 'label':

        print("Ratings")

        break

    elif column == 'id':

        print("ID")

        continue

    else:

        mu = train_df[column].mean()

        sigma = train_df[column].std()

        #mini = train_df[column].min()

        #maxi = train_df[column].max()

        train_df[column] = (train_df[column] - mu)/sigma

        test_df[column] = (test_df[column] - mu)/sigma

        #train_df[column] = (train_df[column] - mini)/(maxi-mini)

        #test_df[column] = (test_df[column] - mini)/(maxi-mini)

                        
X_pred = test_df.iloc[:,1:98]

print(X_pred)
train_df.corr()
train_df.head(14)
Y = train_df.iloc[:,98]

X = train_df.iloc[:,1:98]

print(X)

print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1,shuffle = False)

print(X_train.shape, Y_train.shape)

print(X_test.shape, Y_test.shape)
random_forest_regressor = ensemble.RandomForestRegressor(n_estimators = 100)

model_random_forest_regressor = random_forest_regressor.fit(X,Y)

Y_pred_random_forest_regressor = model_random_forest_regressor.predict(X_test)

print(sqrt(mean_squared_error(Y_pred_random_forest_regressor,Y_test)))
predictions_random_forest_regressor = random_forest_regressor.predict(X_pred)

print(predictions_random_forest_regressor)

plt.plot(predictions_random_forest_regressor);
f = open('solution_random_forest.csv','w')

f.write('id,label')

f.write('\n')

for i in range(0,predictions_random_forest_regressor.shape[0]):

    f.write(str(test_df['id'][i]))

    f.write(',')

    f.write(str(predictions_random_forest_regressor[i]))

    f.write('\n')