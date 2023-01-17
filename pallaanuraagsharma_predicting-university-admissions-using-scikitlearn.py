import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd 

import numpy as np 

import seaborn as sns 

import matplotlib.pyplot as plt 

from jupyterthemes import jtplot

jtplot.style(theme='monokai',context='notebook',ticks=True,grid=False)
admission_df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
admission_df.sample(5)
admission_df = admission_df.drop(columns=['Serial No.'])
admission_df.info()
admission_df.isnull().sum()
from pandas_profiling import ProfileReport

report = ProfileReport(admission_df,title='EDA of Dataset')

report
admission_df.describe()
university_df = admission_df.groupby(by='University_Rating').mean()

university_df
admission_df.hist(bins=30,figsize=(20,20))
sns.pairplot(admission_df)
corr_matrix = admission_df.corr()

sns.heatmap(corr_matrix,annot=True)
from sklearn.model_selection import train_test_split as tts
admission_df.columns
x = admission_df.drop(columns=['Chance of Admit '])

y = admission_df[['Chance of Admit ']]
print('Shape of x: ',x.shape)

print('Shape of y: ',y.shape)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler_x = StandardScaler()

x = scaler_x.fit_transform(x)

x
scaler_y = StandardScaler()

y = scaler_y.fit_transform(y)
x_train,x_test,y_train,y_test = tts(x,y,test_size=0.4,random_state=23)
print('Shape of x_train: ',x_train.shape)

print('Shape of y_train: ',y_train.shape)

print('Shape of x_test: ',x_test.shape)

print('Shape of y_test: ',y_test.shape)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)
from sklearn.metrics import mean_squared_error, accuracy_score

accuracy_lr = lr.score(x_test,y_test)

accuracy_lr
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()

dtr.fit(x_train,y_train)

accuracy_dtr = dtr.score(x_test,y_test)

accuracy_dtr
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100,max_depth=10)

rfr.fit(x_train,y_train)

accuracy_rfr = rfr.score(x_test,y_test)

accuracy_rfr
y_pred_lr = lr.predict(x_test)

plt.plot(y_test,y_pred_lr,'^',color='b')
y_pred_dtr = dtr.predict(x_test)

plt.plot(y_test,y_pred_dtr,'^',color='b')
y_pred_rfr = rfr.predict(x_test)

plt.plot(y_test,y_pred_rfr,'^',color='b')
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

from math import sqrt
y_test_org = scaler_y.inverse_transform(y_test)
y_pred_lr_org = scaler_y.inverse_transform(y_pred_lr)
k = x_test.shape[1]

n = len(x_test)
RMSE   = float(format(np.sqrt(mean_squared_error(y_test_org,y_pred_lr_org)),'.3f'))

MSE    = mean_squared_error(y_test_org,y_pred_lr_org)

MAE    = mean_absolute_error(y_test_org,y_pred_lr_org)

r2     = r2_score(y_test_org,y_pred_lr_org)

adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE = ',RMSE,'MSE = ',MSE,'MAE = ',MAE,'R2 SCORE = ',r2,'ADJ_R2 = ',adj_r2)
y_pred_dtr_org = scaler_y.inverse_transform(y_pred_dtr)
RMSE   = float(format(np.sqrt(mean_squared_error(y_test_org,y_pred_dtr_org)),'.3f'))

MSE    = mean_squared_error(y_test_org,y_pred_dtr_org)

MAE    = mean_absolute_error(y_test_org,y_pred_dtr_org)

r2     = r2_score(y_test_org,y_pred_dtr_org)

adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE = ',RMSE,'MSE = ',MSE,'MAE = ',MAE,'R2 SCORE = ',r2,'ADJ_R2 = ',adj_r2)
y_pred_rfr_org = scaler_y.inverse_transform(y_pred_rfr)
RMSE   = float(format(np.sqrt(mean_squared_error(y_test_org,y_pred_rfr_org)),'.3f'))

MSE    = mean_squared_error(y_test_org,y_pred_rfr_org)

MAE    = mean_absolute_error(y_test_org,y_pred_rfr_org)

r2     = r2_score(y_test_org,y_pred_rfr_org)

adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE = ',RMSE,'MSE = ',MSE,'MAE = ',MAE,'R2 SCORE = ',r2,'ADJ_R2 = ',adj_r2)