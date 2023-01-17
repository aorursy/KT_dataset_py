# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sklearn.metrics
import math
import datetime as dt
%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing data
data=pd.read_csv("/kaggle/input/datasetucimlairquality/AirQualityUCI.csv")
data.shape
data.head()
#converting date and time to datetime
data['Datetime']=data.Date+' '+data.Time
data['Time']=pd.to_datetime(data.Datetime)
data.dtypes
#removing date and datetime 
data.drop('Date', axis=1, inplace=True)
data.drop('Datetime', axis=1, inplace=True)
data.drop('CO_level', axis=1, inplace=True)
#the column seems to have a lot of misplaced data(-200)
data.drop('NMHC_GT', axis=1, inplace=True)
data.head()
#-200 seems to indicate missing data
data.replace(to_replace= -200, value= np.NaN, inplace= True)
def VALUE_CORRECTION(col):
    data[col] = data.groupby('Time')[col].transform(lambda x:x.fillna(x.mean()))
#filing empty spaces with the mean
col_list = data.columns[1:12]

for i in col_list:
    VALUE_CORRECTION(i)
data.info()
data.fillna(method='ffill', inplace= True)
data.info()
#RH vs T
plt.figure(figsize=(25,10))
plt.xlabel('Temperature(°C)')
plt.ylabel('Relative Humidity')
plt.title("Relative Humidity vs Temperature(°C)")
plt.scatter(data['T'], data['RH'], marker='.', aa=True)
#AH vs T
plt.figure(figsize=(25,10))
plt.xlabel('Temperature(°C)')
plt.ylabel('Absolute Humidity')
plt.title("Absolute Humidity vs Temperature(°C)")
plt.scatter(data['T'], data['AH'], marker='.', aa=True)
data.corr() 
#creating linear model
from sklearn.model_selection import train_test_split
X=data[['CO_GT', 'PT08_S1_CO',	'C6H6_GT',	'PT08_S2_NMHC',	'Nox_GT',	'PT08_S3_Nox',	'NO2_GT',	'PT08_S4_NO2',	'PT08_S5_O3',	'T']]
y=data[['RH','AH']]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3)
from sklearn.linear_model import LinearRegression
reg=LinearRegression(normalize="Boolean")
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)
y_pred
y_test
#checking accuracy
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))
reg.intercept_
reg.coef_
#making columns for predicted data
X=data[['CO_GT', 'PT08_S1_CO',	'C6H6_GT',	'PT08_S2_NMHC',	'Nox_GT',	'PT08_S3_Nox',	'NO2_GT',	'PT08_S4_NO2',	'PT08_S5_O3',	'T']]
y_pred_all=reg.predict(X)
data['RH_pred']=y_pred_all[:,0]
data['AH_pred']=y_pred_all[:,1]
data
#sklearn model comes out to be inaccurate at some places and it is quite evidient from the r2 score
#RH vs Datetime
plt.figure(figsize=(22,15))
plt.plot_date(data.Time, data.RH, marker='.', label="True", alpha=0.5)
plt.plot_date(data.Time, data.RH_pred, marker='.', label="Predicted")
plt.title("Comparison of True and Predicted values of Relative Humidity at various times", fontsize=20)
plt.xlabel("Datetime", fontsize=20)
plt.ylabel("Relative Humidity", fontsize=20)
plt.legend(fontsize=15, facecolor='white', markerscale=2)
#AH vs Datetime
plt.figure(figsize=(22,15))
plt.plot_date(data.Time, data.AH, marker='.', label="True", alpha=0.5)
plt.plot_date(data.Time, data.AH_pred, marker='.', label="Predicted")
plt.title("Comparison of True and Predicted values of Absolute Humidity at various times", fontsize=20)
plt.xlabel("Datetime", fontsize=20)
plt.ylabel("Absolute Humidity", fontsize=20)
plt.legend(fontsize=15, facecolor='white', markerscale=2)
#hyperparameter optimisation
n_estimators=[100,200,500,750,1000,1100,1200]
max_depth=[3,5,10,15,20]
booster=['gbtree']
learning_rate=[0.03, 0.06, 0.1, 0.15, 0.2]
min_child_weight=[1,2,3,4]
base_score=[0.2,0.25, 0.5, 0.75]

hyperparameter_grid={'n_estimators':n_estimators,
                     'max_depth':max_depth,
                     'learning_rate':learning_rate,
                     'min_child_weight':min_child_weight,
                     'booster':booster,
                     'base_score':base_score}
import xgboost
xreg=xgboost.XGBRegressor()  #for RH
xreg1=xgboost.XGBRegressor()  #for AH
#for RH
y1=data[['RH']]
X_train, X_test, y1_train, y1_test=train_test_split(X, y1, test_size=0.3)
from sklearn.model_selection import RandomizedSearchCV
#FOR RH
random_cv1=RandomizedSearchCV(estimator=xreg,
                             param_distributions=hyperparameter_grid,
                             n_iter=50,
                             verbose=5,
                             n_jobs=4,
                             scoring='neg_mean_squared_error',
                             return_train_score=True,
                             random_state=42)
random_cv1.fit(X_train, y1_train)
random_cv1.best_estimator_
xreg=xgboost.XGBRegressor(base_score=0.2, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.1, max_delta_step=0, max_depth=5,
             min_child_weight=2, monotone_constraints=None,
             n_estimators=1200, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
             validate_parameters=False, verbosity=None)
xreg.fit(X_train, y1_train)
y1_pred=xreg.predict(X_test)
y1_pred
y1_test
r2_score(y1_test, y1_pred)
#Plotting
data['RH_pred_xg']=xreg.predict(X)
plt.figure(figsize=(22,15))
plt.plot_date(data.Time, data.RH, marker='.', label="True")
plt.plot_date(data.Time, data.RH_pred_xg, marker='.', label="Predicted")
plt.title("Comparison of True and Predicted values of Relative Humidity at various times (with boosting)", fontsize=20)
plt.xlabel("Datetime", fontsize=20)
plt.ylabel("Relative Humidity", fontsize=20)
plt.legend(fontsize=15, facecolor='white', markerscale=2)
#for AH
y2=data[['AH']]
X_train, X_test, y2_train, y2_test=train_test_split(X, y2, test_size=0.3)
xreg1=xgboost.XGBRegressor()
random_cv2=RandomizedSearchCV(estimator=xreg,
                             param_distributions=hyperparameter_grid,
                             n_iter=50,
                             verbose=5,
                             n_jobs=4,
                             scoring='neg_mean_squared_error',
                             return_train_score=True,
                             random_state=42)
random_cv2.fit(X_train, y2_train)
random_cv2.best_estimator_
xreg1=xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.06, max_delta_step=0, max_depth=5,
             min_child_weight=4, monotone_constraints=None,
             n_estimators=1000, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
             validate_parameters=False, verbosity=None)
xreg1.fit(X_train, y2_train)
y2_pred=xreg1.predict(X_test)
y2_pred
y2_test
r2_score(y2_test, y2_pred)
#AH vs Datetime
data['AH_pred_xg']=xreg1.predict(X)
plt.figure(figsize=(22,15))
plt.plot_date(data.Time, data.AH, marker='.', label="True")
plt.plot_date(data.Time, data.AH_pred_xg, marker='.', label="Predicted")
plt.title("Comparison of True and Predicted values of Absolute Humidity at various times (with boosting)", fontsize=20)
plt.xlabel("Datetime", fontsize=20)
plt.ylabel("Absolute Humidity", fontsize=20)
plt.legend(fontsize=15, facecolor='white', markerscale=2)