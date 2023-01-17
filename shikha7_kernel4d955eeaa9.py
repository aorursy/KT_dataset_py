# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

wine=pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

#wine.info()

from sklearn.model_selection import train_test_split

X=wine[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]

y=wine['quality']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)



from sklearn.preprocessing import StandardScaler

scale=StandardScaler().fit(X_train)

X_train_scale=scale.transform(X_train)

X_test_scale=scale.transform(X_test)



#LinearRegg

from sklearn.linear_model import LinearRegression

linreg=LinearRegression().fit(X_train_scale,y_train)

train_pred = linreg.predict(X_train_scale)

test_pred = linreg.predict(X_test_scale) 

print('Linear Regression Score train:{:.2f}'.format(linreg.score(X_train_scale,y_train)))

print('Linear Regression Score test:{:.2f}'.format(linreg.score(X_test_scale,y_test)))

# calculating rmse

from sklearn.metrics import mean_squared_error

train_rmse = mean_squared_error(train_pred, y_train) ** 0.5

print(train_rmse)

test_rmse = mean_squared_error(test_pred, y_test) ** 0.5

print(test_rmse)



predicted_data = np.round_(test_pred)

#predicted_data



#randomforest

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100).fit(X_train_scale,y_train)

train_pred = rf.predict(X_train_scale)

test_pred = rf.predict(X_test_scale) 

print('Random Forest Regression Score train:{:.2f}'.format(rf.score(X_train_scale,y_train)))

print('Random Forest Regression Score test:{:.2f}'.format(rf.score(X_test_scale,y_test)))

# calculating rmse

from sklearn.metrics import mean_squared_error

train_rmse = mean_squared_error(train_pred, y_train) ** 0.5

print(train_rmse)

test_rmse = mean_squared_error(test_pred, y_test) ** 0.5

print(test_rmse)

# calculating mae

from sklearn.metrics import mean_absolute_error

train_mae = mean_absolute_error(train_pred, y_train) ** 0.5

print('MAE:{:.2f}'.format(train_mae))

test_mae = mean_absolute_error(test_pred, y_test) ** 0.5

print('MAE:{:.2f}'.format(test_mae))