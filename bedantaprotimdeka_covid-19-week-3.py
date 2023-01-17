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
import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

%matplotlib inline

train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

display(train_df.info())
display(test_df.info())

display(train_df.head())
display(test_df.head())
display(train_df.isnull().sum())
display(train_df[~train_df['Province_State'].isnull()]['Country_Region'].value_counts())

display(train_df[train_df['Province_State'].isnull()]['Country_Region'].value_counts())

display(train_df['Date'].describe())
show_cum = train_df.groupby(by='Country_Region')[['ConfirmedCases','Fatalities']].max().reset_index()
plt.figure(figsize=(20,10))
#sns.set()
sns.barplot(x='ConfirmedCases',y='Country_Region',data=show_cum[show_cum['ConfirmedCases'] != 0].sort_values(by='ConfirmedCases',ascending=False).head(30))
plt.figure(figsize=(20,10))
sns.barplot(x='Fatalities',y='Country_Region',data=show_cum[show_cum['Fatalities'] != 0].sort_values(by='Fatalities',ascending=False).head(30))
train_df.shape, test_df.shape
train_df['Province_State'] = train_df['Province_State'].fillna('unknown')
test_df['Province_State'] = test_df['Province_State'].fillna('unknown')
test_id = test_df['ForecastId']
train_df.drop(['Id'], axis=1, inplace=True)
test_df.drop('ForecastId', axis=1, inplace=True)
import datetime
train_df['Date'] = pd.to_datetime(train_df['Date'], infer_datetime_format=True)
test_df['Date'] = pd.to_datetime(test_df['Date'], infer_datetime_format=True)
train_df.loc[:, 'Date'] = train_df.Date.dt.strftime("%m%d")
train_df["Date"]  = train_df["Date"].astype(int)
test_df.loc[:, 'Date'] = test_df.Date.dt.strftime("%m%d")
test_df["Date"]  = test_df["Date"].astype(int)
#Lets take our target variable
y_train_cc = train_df['ConfirmedCases']
y_train_ft = train_df['Fatalities']
train_df.drop(['ConfirmedCases'], axis=1, inplace=True)
train_df.drop(['Fatalities'], axis=1, inplace=True)
#Now lets encode the catagorical variable
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
train_df['Country_Region'] = labelencoder.fit_transform(train_df['Country_Region'])
train_df['Province_State'] = labelencoder.fit_transform(train_df['Province_State'])
test_df['Country_Region'] = labelencoder.fit_transform(test_df['Country_Region'])
test_df['Province_State'] = labelencoder.fit_transform(test_df['Province_State'])
x_train = train_df.iloc[:,:].values
x_test = test_df.iloc[:,:].values
from xgboost import XGBRegressor
regressor1 = XGBRegressor(n_estimators = 1000)
regressor1.fit(x_train, y_train_cc)
y_pred_cc= regressor1.predict(x_test)
regressor2 = XGBRegressor(n_estimators = 1000)
regressor2.fit(x_train, y_train_ft)
y_pred_ft= regressor2.predict(x_test)
#Sumbmission the result
df_sub = pd.DataFrame()
df_sub['ForecastId'] = test_id
df_sub['ConfirmedCases'] = y_pred_cc
df_sub['Fatalities'] = y_pred_ft
df_sub.to_csv('submission.csv', index=False)
df_sub.head()