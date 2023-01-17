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

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

train_df.head()
train_df.info()
train_df.rename(columns = {"Id":"ForecastId"},inplace=True)

train_df.sample(5)
train_df['Date'] = pd.to_datetime(train_df['Date'], infer_datetime_format=True)

test_df['Date'] = pd.to_datetime(test_df['Date'], infer_datetime_format=True)
train_df_confirmed  = train_df.where(train_df.ConfirmedCases!=0).dropna(axis=0)

train_df_confirmed.groupby('Country_Region').ConfirmedCases.sum().plot(kind='barh', figsize=(10,5))

plt.show()
train_df_fatalities  = train_df.where(train_df.Fatalities!=0).dropna(axis=0)

train_df_fatalities.groupby('Country_Region').Fatalities.sum().plot(kind='barh', figsize=(10,5))

plt.show()


train_df.head()
test_df.Province_State.fillna(test_df.Country_Region, inplace=True)

train_df.Province_State.fillna(train_df.Country_Region, inplace=True)

X_train = train_df.copy()

X_test = test_df.copy()
from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()

X_train.Country_Region = label_encoder.fit_transform(X_train.Country_Region)

X_train.Province_State = label_encoder.fit_transform(X_train.Province_State)



X_test.Country_Region = label_encoder.fit_transform(X_test.Country_Region)

X_test.Province_State = label_encoder.fit_transform(X_test.Province_State)
y_train_1 = X_train.pop('ConfirmedCases')

y_train_2 = X_train.pop('Fatalities')



X_train.loc[:, 'Date'] = X_train.Date.dt.strftime("%m%d")

X_train["Date"]  = X_train["Date"].astype(int)



X_test.loc[:, 'Date'] = X_test.Date.dt.strftime("%m%d")

X_test["Date"]  = X_test["Date"].astype(int)

X_test_forecast_id = X_test.loc[:, 'ForecastId']
from xgboost import XGBRegressor

cc_model = XGBRegressor(n_estimators=1000)

cc_model.fit(X_train, y_train_1)

y_pred_1 = cc_model.predict(X_test)

y_pred_1 = y_pred_1.round()        
f_model = XGBRegressor(n_estimators=1000)

f_model.fit(X_train, y_train_2)

y_pred_2 = f_model.predict(X_test)

y_pred_2 = y_pred_2.round()     

for i in range(0,len(y_pred_2)):

    if(y_pred_2[i]<0):

        y_pred_2[i] = 0
final_df = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

final_df.ForecastId = final_df.ForecastId.astype('int')

df = pd.DataFrame({'ForecastId': X_test_forecast_id, 'ConfirmedCases': y_pred_1, 'Fatalities': y_pred_2})

submit_df = pd.concat([final_df, df], axis=0)
submit_df.to_csv('submission.csv',index=False)