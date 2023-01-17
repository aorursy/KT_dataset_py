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
#import libraries

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
# read file

data=pd.read_csv("../input/Fish.csv")

data.head()
data.info()
# checking None value

data.isnull().sum()
a=data.iloc[:,2:]

a.head()
b=data.iloc[:,1]

b.head()
data['Species'].value_counts()
important_data=['Species','Length1','Length2','Length3','Height','Width']

X=data[important_data]

y=data.Weight

print(X.head())

print(y.head())
#label encoding

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder = LabelEncoder()

X.iloc[:,0] = label_encoder.fit_transform(X.iloc[:,0]) #LabelEncoder is used to encode the country value

# one hot encoder

hot_encoder = OneHotEncoder(categorical_features = [0])

X = hot_encoder.fit_transform(X).toarray()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)
# Fitting MLR to the Training Set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
# predict value 

y_pred = regressor.predict(X_test)
y_pred
(abs(y_test-y_pred)).describe()
regressor.score(X_train, y_train)
#RMSE ( root mean square error)

#r^2 ( r- square)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))

r2 = r2_score(y_test,y_pred)

print('rmse:',rmse)

print('r2:',r2)
data.corr()