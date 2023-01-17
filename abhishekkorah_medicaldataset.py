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
import pandas as pd
data = pd.read_csv('/kaggle/input/insurance/insurance.csv')
data.head()
data.isnull().sum()
data.describe().transpose()
x = data.loc[:,['age', 'bmi']]
y = data.charges
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train,y_train)
print(reg.intercept_)
print(reg.coef_)
LR_train_y_pred = reg.predict(x_train)
import numpy as np
from sklearn import metrics

print("Training :- ")
print('MAE :- ', metrics.mean_absolute_error(y_train,LR_train_y_pred))
print('MSE :- ', metrics.mean_squared_error(y_train,LR_train_y_pred))
print('RMSE :- ', np.sqrt(metrics.mean_squared_error(y_train,LR_train_y_pred)))
print('Coefficient of Determination')
print('R^2 :- ', metrics.r2_score(y_train,LR_train_y_pred))
LR_test_y_pred = reg.predict(x_test)
print("Testing :- ")
print('MAE :- ', metrics.mean_absolute_error(y_test,LR_test_y_pred))
print('MSE :- ', metrics.mean_squared_error(y_test,LR_test_y_pred))
print('RMSE :- ', np.sqrt(metrics.mean_squared_error(y_test,LR_test_y_pred)))
print('Coefficient of Determination')
print('R^2 :- ', metrics.r2_score(y_test,LR_test_y_pred))
from sklearn.tree import DecisionTreeRegressor

D_reg_model = DecisionTreeRegressor()
D_reg_model.fit(x_train,y_train)
DTR_train_y_pred = D_reg_model.predict(x_train)
print("Training :- ")

print('MAE :- ', metrics.mean_absolute_error(y_train,DTR_train_y_pred))
print('MSE :- ', metrics.mean_squared_error(y_train,DTR_train_y_pred))
print('RMSE :- ', np.sqrt(metrics.mean_squared_error(y_train,DTR_train_y_pred)))
print('Coefficient of Determination')
print('R^2 :- ', metrics.r2_score(y_train,DTR_train_y_pred))
DTR_test_y_pred = D_reg_model.predict(x_test)
print("Testing :- ")

print('MAE :- ', metrics.mean_absolute_error(y_test,DTR_test_y_pred))
print('MSE :- ', metrics.mean_squared_error(y_test,DTR_test_y_pred))
print('RMSE :- ', np.sqrt(metrics.mean_squared_error(y_test,DTR_test_y_pred)))
print('Coefficient of Determination')
print('R^2 :- ', metrics.r2_score(y_test,DTR_test_y_pred))
from sklearn.ensemble import RandomForestRegressor

R_reg_model = RandomForestRegressor()
R_reg_model.fit(x_train,y_train)
RFR_train_y_pred = R_reg_model.predict(x_train)
print("Training :- ")

print('MAE :- ', metrics.mean_absolute_error(y_train,RFR_train_y_pred))
print('MSE :- ', metrics.mean_squared_error(y_train,RFR_train_y_pred))
print('RMSE :- ', np.sqrt(metrics.mean_squared_error(y_train,RFR_train_y_pred)))
RFR_train_y_pred
print('Coefficient of Determination')
print('R^2 :- ', metrics.r2_score(y_train,RFR_train_y_pred))
RFR_test_y_pred = R_reg_model.predict(x_test)
print("Testing :- ")

print('MAE :- ', metrics.mean_absolute_error(y_test,RFR_test_y_pred))
print('MSE :- ', metrics.mean_squared_error(y_test,RFR_test_y_pred))
print('RMSE :- ', np.sqrt(metrics.mean_squared_error(y_test,RFR_test_y_pred)))

print('Coefficient of Determination')
print('R^2 :- ', metrics.r2_score(y_test,RFR_test_y_pred))
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(x_train, y_train)
print(lasso_model.coef_)
print(lasso_model.intercept_)
lasso_train_y_pred = lasso_model.predict(x_train)
print("Training :- ")

print('MAE :- ', metrics.mean_absolute_error(y_train,lasso_train_y_pred))
print('MSE :- ', metrics.mean_squared_error(y_train,lasso_train_y_pred))
print('RMSE :- ', np.sqrt(metrics.mean_squared_error(y_train,lasso_train_y_pred)))

print('Coefficient of Determination')
print('R^2 :- ', metrics.r2_score(y_train,lasso_train_y_pred))
lasso_test_y_pred = lasso_model.predict(x_test)
print("Training :- ")

print('MAE :- ', metrics.mean_absolute_error(y_test,lasso_test_y_pred))
print('MSE :- ', metrics.mean_squared_error(y_test,lasso_test_y_pred))
print('RMSE :- ', np.sqrt(metrics.mean_squared_error(y_test,lasso_test_y_pred)))

print('Coefficient of Determination')
print('R^2 :- ', metrics.r2_score(y_test,lasso_test_y_pred))
