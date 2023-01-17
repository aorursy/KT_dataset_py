# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler as sc

names = ('Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP',

       'LOR ', 'CGPA', 'Research', 'Chance of Admit ')

data = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv', skiprows = 400, names = names)

data.head()
data.info()
data.columns
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor 

from sklearn.ensemble import AdaBoostRegressor



X=data.drop(['Chance of Admit ', 'Serial No.'], axis=1)

y = data['Chance of Admit ']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm

y_pred = regressor.predict(X_test)

RMSE_Linear = np.sqrt(metrics.mean_squared_error(y_test, y_pred))



print('Mean Absolute Error_lng:', metrics.mean_absolute_error(y_test, y_pred).round(3))  

print('Mean Squared Error_lng:', metrics.mean_squared_error(y_test, y_pred).round(3))  

print('Root Mean Squared Error_lng:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))

print('r2_score_lng:', r2_score(y_test, y_pred).round(3))
ridge = Ridge(alpha=1.0)

ridge.fit(X_train, y_train) #training the algorithm



y_pred = ridge.predict(X_test)

RMSE_Ridge = np.sqrt(metrics.mean_squared_error(y_test, y_pred))



print('Mean Absolute Error_ridge:', metrics.mean_absolute_error(y_test, y_pred).round(3))  

print('Mean Squared Error_ridge:', metrics.mean_squared_error(y_test, y_pred).round(3))  

print('Root Mean Squared Error_ridge:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))

print('r2_score_ridge:', r2_score(y_test, y_pred).round(3))
clf = Lasso(alpha=0.1)



clf.fit(X_train, y_train) #training the algorithm



y_pred = clf.predict(X_test)

RMSE_Lasso = np.sqrt(metrics.mean_squared_error(y_test, y_pred))



print('Mean Absolute Error_lasso:', metrics.mean_absolute_error(y_test, y_pred).round(3))  

print('Mean Squared Error_lasso:', metrics.mean_squared_error(y_test, y_pred).round(3))  

print('Root Mean Squared Error_lasso:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))

print('r2_score_lasso:', r2_score(y_test, y_pred).round(3))

rfe = RandomForestRegressor(n_estimators = 100, random_state = 42) 

  

# fit the regressor with x and y data 

rfe.fit(X, y)   

y_pred=rfe.predict(X_test)

RMSE_RFE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('r2_score_RFE:', r2_score(y_test, y_pred).round(3))
ABR = AdaBoostRegressor(n_estimators = 100, random_state = 42) 

  

# fit the regressor with x and y data 

ABR.fit(X, y)   

y_pred=ABR.predict(X_test)

RMSE_ABR = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('r2_score_ABR:', r2_score(y_test, y_pred).round(3))
submission = pd.DataFrame([RMSE_RFE, RMSE_Linear, RMSE_ABR, RMSE_Lasso, RMSE_Ridge], index=['RMSE_RFE', 'RMSE_Linear', 'RMSE_ABR', 'RMSE_Lasso', 

                                                                                            'RMSE_Ridge'],columns = ['RMSE_Score'])

submission.to_csv('result.csv')