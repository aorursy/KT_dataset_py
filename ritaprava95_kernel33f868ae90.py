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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer, StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge





#importing the train and test data into dataframes

train = pd.read_csv("train.csv", index_col=0)

test = pd.read_csv("test.csv", index_col=0)

indexcol = test.index

#spliting the train dataset in to feature and target variables

train_X = train.drop(['SalePrice'], axis=1)

train_y = train.loc[:,'SalePrice']



#getting rid of all the non numeric columns

train_X = pd.get_dummies(train_X)

test = pd.get_dummies(test)



#eliminating the irrelivent column wrt the test data

train_X = train_X.loc[:,list(test.columns)]# eliminating the irrelivent column wrt the test data



#imputing to overcome the problem of missing values

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

train_X = imp.fit_transform(train_X)

test = imp.fit_transform(test)



#scaling the data

scaler = StandardScaler()

train_X = scaler.fit_transform(train_X)

test = scaler.fit_transform(test)



#fitting the traing dataset to the model and predicting the prices from the testset

alpha_space = np.logspace(-4,1,50)

for a in alpha_space:

    ridge = Ridge(alpha=a, normalize=True)

    ridge_cv_scores = cross_val_score(ridge,train_X,train_y,cv=10)

    print(a," :"+str(np.mean(ridge_cv_scores)))



ridge = Ridge(alpha=0.9540954763499944, normalize=True)

ridge.fit(train_X, train_y)

predicted_price = ridge.predict(test)



predicted_df = pd.DataFrame(columns=['SalePrice'],

                            index = indexcol,

                            data=predicted_price)



predicted_df.to_csv('predicted_9.csv')