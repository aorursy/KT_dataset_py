import pandas as pd

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split

import numpy as np

import datetime

import requests

import warnings

import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA

from fbprophet import Prophet

from fbprophet.plot import plot_plotly

warnings.filterwarnings('ignore')

%matplotlib inline
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')



train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])
df1 = confirmed_df.groupby('Country/Region').sum().reset_index()

df2 = deaths_df.groupby('Country/Region').sum().reset_index()

df3 = recovered_df.groupby('Country/Region').sum().reset_index()

countries = ['China','US', 'Italy', 'Spain', 'France','India']

global_confirmed = []

for country in countries:

    k =df1[df1['Country/Region'] == country].loc[:,'1/30/20':]

    global_confirmed.append(k.values.tolist()[0])

dates = list(confirmed_df.columns[4:])

dates = list(pd.to_datetime(dates))

dates_india = dates[8:]
k =df1[df1['Country/Region']=='India'].loc[:,'2/4/20':]

india_confirmed = k.values.tolist()[0]

growth_diff = []

for i in range(1,len(india_confirmed)):

    growth_diff.append(india_confirmed[i] / india_confirmed[i-1])

growth_factor = sum(growth_diff)/len(growth_diff)

print('Average growth factor',growth_factor)
prediction_dates = []



start_date = dates_india[len(dates_india) - 1]

for i in range(15):

    date = start_date + datetime.timedelta(days=1)

    prediction_dates.append(date)

    start_date = date

previous_day_cases = global_confirmed[5][len(dates_india) - 1]

predicted_cases = []
for i in range(15):

    predicted_value = previous_day_cases *  growth_factor

    predicted_cases.append(predicted_value)

    previous_day_cases = predicted_value
plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 11)

plt.yticks(fontsize = 10)

plt.xlabel("Dates",fontsize = 20)

plt.ylabel('Total cases',fontsize = 20)

plt.title("Predicted Values for the next 15 Days" , fontsize = 20)

ax1 = plt.plot_date(y= predicted_cases,x= prediction_dates,linestyle ='-',color = 'c')
dates = list(confirmed_df.columns[4:])

dates = list(pd.to_datetime(dates))

dates_india = dates[8:]

df1 = confirmed_df.groupby('Country/Region').sum().reset_index()

k = df1[df1['Country/Region']=='India'].loc[:,'1/22/20':]

india_confirmed = k.values.tolist()[0] 

data = pd.DataFrame(columns = ['ds','y'])

data['ds'] = dates

data['y'] = india_confirmed
arima = ARIMA(data['y'], order=(5, 1, 0))

arima = arima.fit(trend='c', full_output=True, disp=True)

forecast = arima.forecast(steps= 30)

pred = list(forecast[0])



start_date = data['ds'].max()

prediction_dates = []

for i in range(30):

    date = start_date + datetime.timedelta(days=1)

    prediction_dates.append(date)

    start_date = date

plt.figure(figsize= (15,10))

plt.xlabel("Dates",fontsize = 20)

plt.ylabel('Total cases',fontsize = 20)

plt.title("Predicted Values for the next 15 Days" , fontsize = 20)



plt.plot_date(y= pred,x= prediction_dates,linestyle ='dashed',color = '#ff9999',label = 'Predicted');

plt.plot_date(y=data['y'],x=data['ds'],linestyle = '-',color = 'blue',label = 'Actual');

plt.legend();
confirmedcases_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

df1 = confirmedcases_df.groupby('Country/Region').sum().reset_index()

df1

k = df1[df1['Country/Region']=='India'].loc[:,'1/22/20':]

confirmedcases_India = k.values.tolist()[0]

confirmedcases_India
data=pd.DataFrame(columns=['ds','y'])

dates = list(confirmedcases_df.columns[4:])

dates = list(pd.to_datetime(dates))

dates
data['ds'] = dates

data['y'] = confirmedcases_India

data
prop= Prophet()

prop.fit(data)

future = prop.make_future_dataframe(periods=30)

prop_forecast = prop.predict(future)

forecast = prop_forecast[['ds','yhat']].tail(30)

forecast
fig = plot_plotly(prop, prop_forecast)

fig = prop.plot(prop_forecast,xlabel='Date',ylabel='Confirmed Cases')
train['day'] = train['Date'].dt.day

train['month'] = train['Date'].dt.month

train['dayofweek'] = train['Date'].dt.dayofweek

train['dayofyear'] = train['Date'].dt.dayofyear

train['quarter'] = train['Date'].dt.quarter

train['weekofyear'] = train['Date'].dt.weekofyear

test['day'] = test['Date'].dt.day

test['month'] = test['Date'].dt.month

test['dayofweek'] = test['Date'].dt.dayofweek

test['dayofyear'] = test['Date'].dt.dayofyear

test['quarter'] = test['Date'].dt.quarter

test['weekofyear'] = test['Date'].dt.weekofyear

countries = list(train['Country_Region'].unique())

india_code = countries.index('India')

train = train.drop(['Date','Id'],1)

test =  test.drop(['Date'],1)



train.Province_State.fillna('NaN', inplace=True)

oe = OrdinalEncoder()

train[['Province_State','Country_Region']] = oe.fit_transform(train.loc[:,['Province_State','Country_Region']])



test.Province_State.fillna('NaN', inplace=True)

oe = OrdinalEncoder()

test[['Province_State','Country_Region']] = oe.fit_transform(test.loc[:,['Province_State','Country_Region']])
columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region','ConfirmedCases','Fatalities']

test_columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State','Country_Region']

train = train[columns]

x = train.drop(['Fatalities','ConfirmedCases'], 1)

y = train['ConfirmedCases']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

test = test[test_columns]

test_india = test[test['Country_Region'] == india_code]
models = []

mse = []

mae = []

rmse = []
lgbm = LGBMRegressor(n_estimators=1300)

lgbm.fit(x_train,y_train)

pred = lgbm.predict(x_test)

lgbm_forecast = lgbm.predict(test_india)

models.append('LGBM')

mse.append(round(mean_squared_error(pred, y_test),2))

mae.append(round(mean_absolute_error(pred, y_test),2))

rmse.append(round(np.sqrt(mean_squared_error(pred, y_test)),2))
rf = RandomForestRegressor(n_estimators=100)

rf.fit(x_train,y_train)

pred = rf.predict(x_test)

rfr_forecast = rf.predict(test_india)

models.append('Random Forest')

mse.append(round(mean_squared_error(pred, y_test),2))

mae.append(round(mean_absolute_error(pred, y_test),2))

rmse.append(round(np.sqrt(mean_squared_error(pred, y_test)),2))
xgb = XGBRegressor(n_estimators=100)

xgb.fit(x_train,y_train)

pred = xgb.predict(x_test)

xgb_forecast = xgb.predict(test_india)

models.append('XGBoost')

mse.append(round(mean_squared_error(pred, y_test),3))

mae.append(round(mean_absolute_error(pred, y_test),3))

rmse.append(round(np.sqrt(mean_squared_error(pred, y_test)),3))
print(models)

print(mse)

print(mae)

print(rmse)