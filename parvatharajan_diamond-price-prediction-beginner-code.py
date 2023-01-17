import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')
df.shape
df.head()
del df['Unnamed: 0']
df.head()
df['cut'].value_counts()
df['cut'].replace(('Fair','Good','Very Good','Premium','Ideal'),(1,2,3,4,5), inplace = True)
df['color'].value_counts()
df['color'].replace(('D','E','F','G','H','I','J'),(7,6,5,4,3,2,1), inplace = True)
df['clarity'].replace(('I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'),(1,2,3,4,5,6,7,8),inplace = True)
df.head()
df.isnull().sum()
df.dtypes
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import SelectFromModel

import xgboost as xgb

from sklearn.metrics import mean_squared_error, r2_score
y = df['price']

X = df.drop('price', axis = 1)

print(y.shape)

print(X.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)
print(X_train.shape)

print(X_test.shape)
rf = RandomForestRegressor()

rf.fit(X_train,y_train)
for feature in zip(X. columns, rf.feature_importances_):

    print(feature)
lr = LinearRegression()

lr.fit(X_train,y_train)

pred_lr = lr.predict(X_test)
pred_rf = rf.predict(X_test)
lr_mse = mean_squared_error(y_test,pred_lr)

rf_mse = mean_squared_error(y_test,pred_rf)

print(lr_mse)

print(rf_mse)
print(r2_score(y_test,pred_lr))

print(r2_score(y_test,pred_rf))