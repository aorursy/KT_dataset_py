# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.





train_data = pd.read_csv('../input/train.csv') #Importing train data

test_data = pd.read_csv('../input/test.csv') #Importing test data
#checking data

train_data.describe()

test_data.head()
from sklearn.model_selection import train_test_split



#print(train_data.shape)

print (train_data.columns)

Pclass_OneHotEncoding = pd.get_dummies(train_data.Pclass)

sex_OneHotEncoding = pd.get_dummies(train_data.Sex)

features = ['Sex', 'Age', 'Pclass']



X = pd.get_dummies(train_data[features],columns =['Pclass', 'Sex'])

y = train_data['Survived']



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
from xgboost import XGBRegressor





ML_xboost =   XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth = 6)



# fit rf_model_on_full_data on all data from the training data

ML_xboost.fit(train_X, train_y, early_stopping_rounds=5, 

             eval_set=[(val_X, val_y)], verbose=False)


test_X = pd.get_dummies(test_data[features],columns =['Pclass', 'Sex'])

#print (test_data.iloc[2])

#print (test_X.head())

test_preds = ML_xboost.predict(test_X)

#print(test_preds[:50])

test_preds = [ 0 if value<0.5 else 1 for value in test_preds]

#print(test_preds[:50])



output = pd.DataFrame({'PassengerId': test_data.PassengerId,

                       'Survived': test_preds})

output.to_csv('mySubmission_fix.csv', index=False)

print ('Done')




