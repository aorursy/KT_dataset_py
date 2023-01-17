# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
#importing the dataset

data=pd.read_csv('../input/price-prediction/50_Startups.csv')

X=data.iloc[:,:-1].values

y=data.iloc[:,-1].values
print(X)
#Encoding the categorical data by one hot encoding

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')

X=np.array(ct.fit_transform(X))
print(X)
#splitting the dataset into test and train data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=20,random_state=0)
#training the Multiple linear regression model on the training set

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,y_train)
#predicting the test set results

y_pred=regressor.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
print("Training data accuracy:")

regressor.score(X_train,y_train)

print("Test data accuracy:")

regressor.score(X_test,y_test)