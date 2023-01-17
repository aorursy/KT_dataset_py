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
# Importing necessary packages
import numpy as np
import pandas as pd
import math as m
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sb
import datetime as dt
from itertools import product
from collections import Counter
from matplotlib.pylab import rcParams
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from statistics import stdev
import re
from category_encoders import *
from datetime import timedelta
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.stats import norm

# Importing SARIMA packages

from statsmodels.tsa.statespace.sarimax import SARIMAX
# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')

# To visualise all the columns in a dataframe
pd.pandas.set_option('display.max_columns', None)

# Setting maximum row numbers
pd.set_option('display.max_rows', 1000)
# Importing the dataset

data_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
data_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

print(data_train.shape)
print(data_test.shape)
data_train.head()
data_train.info()
# Checking for Null values
data_train.isnull().sum()
# Converting feature 'Date' into datetime format

data_train['Date'] = pd.to_datetime(data_train['Date'])

# Creating new feature column - 'Countries_Province'
data_train['Countries_Province'] = data_train['Country_Region']+data_train['Province_State'].fillna('')
data_test['Countries_Province'] = data_test['Country_Region']+data_test['Province_State'].fillna('')

print(data_train.shape)
print(data_test.shape)
data_train.head()
# Basic EDA

print('Unique count of ID :',len(data_train['Id'].unique()))
print('\n')
print('Unique count of Country_Region :',len(data_train['Country_Region'].unique()))
print('\n')
print('Minimum Date :',data_train['Date'].min())
print('\n')
print('Maximum Date :',data_train['Date'].max())
print('\n')
print("Unique count of Countries_Province :",len(data_train['Countries_Province'].unique()))
# Dropping feature Id and Province_State

data_train.drop(['Id','Province_State'],axis = 1,inplace = True)

print(data_train.shape)
# Checking distribution along 25%,50%,75% percentiles
data_train.describe()
# Data Visualization -- Worldwide

d = data_train.groupby(data_train['Date']).sum()
d.plot(figsize = (15,6));
plt.ylabel('Sum');
plt.title('Summary of COVID 19 cases Worldwide');
# Data Visualization -- US

df_usa = data_train.loc[data_train['Country_Region']== 'US']
df_usa = df_usa.groupby(df_usa['Date']).sum()
df_usa.plot(figsize = (15,6));
plt.ylabel('Sum');
plt.title('Summary of COVID 19 cases for US');
# Data Visualization -- Italy

df_ita = data_train.loc[data_train['Country_Region']== 'Italy']
df_ita = df_ita.groupby(df_ita['Date']).sum()
df_ita.plot(figsize = (15,6));
plt.ylabel('Sum');
plt.title('Summary of COVID 19 cases for Italy');
# Data Visualization -- Spain

df_spain =  data_train[data_train.Country_Region=='Spain'].groupby('Date')['ConfirmedCases','Fatalities'].sum()
df_spain.plot(figsize = (15,6));
plt.ylabel('Sum');
plt.title('Summary of COVID 19 cases for Spain');
# Data Visualization -- India

df_ind = data_train.loc[data_train['Country_Region']== 'India']
df_ind = df_ind.groupby(df_ind['Date']).sum()
df_ind.plot(figsize = (15,6));
plt.ylabel('Sum');
plt.title('Summary of COVID 19 cases for India');
# Histogram plot of Confirmed cases worldwide
plt.figure(figsize=(15,6));
plt.xticks(rotation = 30)
sb.distplot(d['ConfirmedCases'],kde = False,color="g");
# Histogram plot of Fatalities cases worldwide
plt.figure(figsize=(15,6));
plt.xticks(rotation = 30)
sb.distplot(d['Fatalities'],kde = False,color="b");
# Stationarity Check
from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') 
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")

# Dickey Fuller test for feature 'ConfirmedCases'
adf_test(data_train['ConfirmedCases'])
# Dickey Fuller test for feature 'Fatalities'
adf_test(data_train['Fatalities'])
# Setting the index

data_train.index = data_train['Date']

start_date = '2020-04-02'
end_date = '2020-05-14'

countries_pr = data_train['Countries_Province'].unique()

# Training & Prediction

column_names = ['Country','ConfirmedCases','Fatalities']
data_predict = pd.DataFrame(columns = column_names)
data_result = pd.DataFrame()


for value in countries_pr:
    #print("Country :",value)
    
    data_sarima = data_train[(data_train['Countries_Province'] == value)]
    sarima1 = SARIMAX(data_sarima['ConfirmedCases'],order=(2,1,0),freq = 'D',enforce_stationarity=False, enforce_invertibility=False).fit()                       
    sarima2 = SARIMAX(data_sarima['Fatalities'],order=(2,1,0),freq = 'D',enforce_stationarity=False, enforce_invertibility=False).fit()                        
    
    pred1 = sarima1.predict(start_date,end_date) 
    #pred1 = (np.exp(pred1)-1)
    pred2 = sarima2.predict(start_date,end_date)
    #pred2 = (np.exp(pred2)-1)
    
    data_predict['ConfirmedCases'] = round(pred1)
    
    data_predict['Fatalities'] = round(pred2)
    
    data_predict['Country'] = value
    
    data_result = data_result.append(data_predict,sort = True)
    
    data_predict = data_predict[0:0]  # Resetting the dataframe
    
#print('Prediction for Confirmed Cases :',pred1)
#print('Prediction for Fatalities :',pred2)
print('Shape of the dataframe :',data_result.shape)
# Submissions

data_result['ForecastId'] = range(1,13460)

df = data_result[['ForecastId','ConfirmedCases','Fatalities']]

df.to_csv('submission.csv',index = False)
