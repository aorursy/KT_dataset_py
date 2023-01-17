# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from IPython.display import display

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
X_train = pd.read_csv('../input/train.csv')

X_train.head()
# drop columns where 20% of data is Nan.

thresh = len(X_train) * .8

X_train.dropna(thresh = thresh, axis = 1, inplace = True)
# fill na with mean values

X_train.fillna(X_train.mean(), inplace=True)
# one-hot encoding

X_train2 = pd.get_dummies(X_train,drop_first=True)

X_train2.head()
Y_train = X_train2.SalePrice

X_train2.drop(['SalePrice','Id'],axis=1,inplace = True)

print(X_train2.shape)
regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)

regressor.fit(X_train2,Y_train)
# Read test data

X_test = pd.read_csv('../input/test.csv')



# drop 

thresh = len(X_test) * .8

X_test.dropna(thresh = thresh, axis = 1, inplace = True)

# fill na with mean values

X_test.fillna(X_test.mean(), inplace=True)

# one-hot encoding

X_test2 = pd.get_dummies(X_test,drop_first=True)

# remove Id

X_test2.drop('Id',axis=1,inplace = True)

print(X_test2.shape)

X_test2.head()
# One-hot encoding might not capture all params from training set so find missing and set them to zero.

for column in X_train2:

    if column not in X_test2:

        print(column)

        X_test2[column] = 0
X_test2.shape
predicted_prices = regressor.predict(X_test2)

print(predicted_prices)
test = pd.read_csv('../input/test.csv')

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)

print(len(predicted_prices))