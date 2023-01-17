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
data = pd.read_csv('../input/Admission_Predict.csv')

data.describe()
data.columns
y = data['Chance of Admit ']

y.head(5)
X = data.iloc[:,:-1]

print(X.columns)



#describing X dataframe

X.describe()
#correlation matrix

X.corr()
#importing xgboost algorithm 

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.preprocessing import Imputer



#splitting data into training and testing set

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)





#perform imputing for xgboost algorithm

my_imputer = Imputer()

train_X = my_imputer.fit_transform(train_X)

test_X = my_imputer.transform(test_X)



#making model using xgbregressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

my_model.fit(train_X, train_y, early_stopping_rounds=10, 

             eval_set=[(test_X, test_y)], verbose=False)
#using mean squared error predicting the accuracy of the model

from sklearn.metrics import mean_squared_error

predicted_values = my_model.predict(test_X)

print(mean_squared_error(test_y,predicted_values))
#score on test set

print("XGboostregressor'score on the dataset: {}".format(my_model.score(test_X,test_y)*100))
#score on training set

print("XGboostregressor'score on the dataset: {}".format(my_model.score(train_X,train_y)*100))    