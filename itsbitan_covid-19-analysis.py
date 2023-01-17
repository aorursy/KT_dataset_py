# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

df_sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
df_train.info()
df_test.info()
df_sub.info()
df_train.shape, df_test.shape, df_sub.shape
df_train.head()
df_train.drop(['Province_State'], axis=1, inplace=True)

df_test.drop(['Province_State'], axis=1, inplace=True)
test_id = df_test['ForecastId']

df_train.drop(['Id'], axis=1, inplace=True)

df_test.drop('ForecastId', axis=1, inplace=True)
df_train[["ConfirmedCases","Fatalities"]] =df_train[["ConfirmedCases","Fatalities"]].astype(int)
import datetime

df_train['Date'] =df_train['Date'].apply(pd.to_datetime)

df_test['Date'] =df_test['Date'].apply(pd.to_datetime)
# Group dataset by 'Date' with sum parameter and analyse the 'Confirmed','Deaths' values.

cases = df_train.groupby('Date').sum()[['ConfirmedCases', 'Fatalities']]

sns.set(style = 'whitegrid')

cases.plot(kind='line', figsize = (15,7) , marker='o',linewidth=2)

plt.bar(cases.index, cases['ConfirmedCases'],alpha=0.3,color='g')

plt.xlabel('Days', fontsize=15)

plt.ylabel('Number of cases', fontsize=15)

plt.title('Worldwide Covid-19 cases - Confirmed & Fatalities',fontsize=20)

plt.legend()

plt.show()
dates_train = df_train['Date'] 

dates_test = df_test['Date'] 
days_since_1_22 = np.array([i for i in range(len(dates_train))]).reshape(-1, 1)

days_test_since_1_22 = np.array([i for i in range(len(dates_test))]).reshape(-1, 1)
x_train = days_since_1_22

y_train_cc = df_train['ConfirmedCases']
x_test = days_test_since_1_22
# Fitting Polynomial Regression to the dataset

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline
model = make_pipeline(PolynomialFeatures(5), BayesianRidge())

model.fit(x_train,y_train_cc)                

y_pred_cc = model.predict(x_test)
x_train = days_since_1_22

y_train_ft = df_train['Fatalities']
model = make_pipeline(PolynomialFeatures(2), BayesianRidge())

model.fit(x_train,y_train_ft)                

y_pred_ft = model.predict(x_test)
#Sumbmission the result

df_sub = pd.DataFrame()

df_sub['ForecastId'] = test_id

df_sub['ConfirmedCases'] = y_pred_cc

df_sub['Fatalities'] = y_pred_ft

df_sub.to_csv('submission.csv', index=False)
df_sub[["ConfirmedCases","Fatalities"]] =df_sub[["ConfirmedCases","Fatalities"]].astype(int)
df_sub.head()