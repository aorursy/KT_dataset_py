import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





df_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

df_test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

df_submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")
df_train['Date'] = df_train['Date'].apply(pd.to_datetime)

Global_Top_Cases = df_train[df_train['Date'] == df_train['Date'].max()].groupby(['Country_Region','Date'])

G_cases = Global_Top_Cases.sum().sort_values(['ConfirmedCases'], ascending = False)

G_cases[:10]
G1 = G_cases.drop(['Id'], axis = 1)

G1[:10]
%matplotlib inline

G1[:10].plot(kind = 'bar')
G1['ConfirmedCases_percent_change'] = G1['ConfirmedCases'].pct_change()

G1['Fatalities_percent_change'] = G1['Fatalities'].pct_change()

G2 = G1.drop(['ConfirmedCases', 'Fatalities'], axis = 1)

G2[:10].plot(kind = 'bar')

us = df_train[df_train['Country_Region'] == 'US'].groupby(['Province_State']).max()

us_cases = us.sort_values(['ConfirmedCases'], ascending = False)

us_cases.head()
us_cases = us_cases.drop(['Id', 'Date','Country_Region'], axis = 1)

us_cases[:10].plot(kind = 'bar')
us_cases['ConfirmedCases_percent_change'] = us_cases['ConfirmedCases'].pct_change()

us_cases['Fatalities_percent_change'] = us_cases['Fatalities'].pct_change()

us_cases = us_cases.drop(['ConfirmedCases', 'Fatalities'], axis = 1)

us_cases[:10].plot(kind = 'bar')
df_train = df_train[df_train['Date'] < df_test.Date.min() ]

df_train['Country_Region'] = df_train['Country_Region'].astype('category')

df_test['Country_Region'] = df_test['Country_Region'].astype('category')

df_train["Country_Region_cat"] = df_train["Country_Region"].cat.codes

df_test["Country_Region_cat"] = df_test["Country_Region"].cat.codes

df_train['date_updated'] = df_train['Date'].apply(pd.to_datetime)

df_test['date_updated'] = df_test['Date'].apply(pd.to_datetime)
def createDateFields(df):

    df['year'] = df['date_updated'].dt.year 

    df['month'] = df['date_updated'].dt.month 

    df['day'] = df['date_updated'].dt.day 

    #df_train['hour'] = df_train['date_updated'].dt.hour 

    #df_train['minute'] = df_train['date_updated'].dt.minute 

    return df

df_train = createDateFields(df_train)

df_test = createDateFields(df_test)

df_train.set_index('date_updated', inplace = True)

df_test.set_index('date_updated', inplace = True)

df_train.drop(['Id','Province_State', 'Date'], axis =1)

df_test.drop(['ForecastId','Province_State', 'Date'], axis =1)
import xgboost as xgb

features = ['Country_Region_cat',   'year', 'month', 'day']

X = df_train[features]

y1 = df_train['ConfirmedCases']

y2 = df_train['Fatalities']

data_dmatrix1 = xgb.DMatrix(data=X,label=y1)

data_dmatrix2 = xgb.DMatrix(data=X,label=y2)

xg_reg_model1 = xgb.XGBRegressor(n_estimators = 1000)

xg_reg_model1.fit(X,y1);

xg_reg_model2 = xgb.XGBRegressor(n_estimators = 1000)



xg_reg_model2.fit(X,y2);

test_features = ['Country_Region_cat',  'year', 'month', 'day']

X_test = df_test[test_features]

y_pred1 = xg_reg_model1.predict(data = X_test)

y_pred2 = xg_reg_model2.predict(data = X_test)

submission = pd.DataFrame()

submission["ForecastId"] = df_submission.ForecastId

submission["ConfirmedCases"] = y_pred1.astype(int).clip(min=0)

submission["Fatalities"] = y_pred2.astype(int).clip(min=0)

submission.head()
submission.to_csv("submission.csv",index = False)