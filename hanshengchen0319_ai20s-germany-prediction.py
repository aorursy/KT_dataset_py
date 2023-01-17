import pandas as pd

import numpy as np

import datetime

import requests

import warnings



import matplotlib.pyplot as plt

import matplotlib

import matplotlib.dates as mdates

import seaborn as sns

import squarify

import plotly.offline as py

import plotly_express as px



from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.preprocessing import OrdinalEncoder

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

from statsmodels.tsa.arima_model import ARIMA

from fbprophet import Prophet

from fbprophet.plot import plot_plotly, add_changepoints_to_plot



from IPython.display import Image

warnings.filterwarnings('ignore')

%matplotlib inline



age_details = pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')

india_covid_19 = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

hospital_beds = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')

individual_details = pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')

ICMR_labs = pd.read_csv('../input/covid19-in-india/ICMRTestingLabs.csv')

state_testing = pd.read_csv('../input/covid19-in-india/StatewiseTestingDetails.csv')

population = pd.read_csv('../input/covid19-in-india/population_india_census2011.csv')



world_population = pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv')

confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-04-2020.csv')



india_covid_19['Date'] = pd.to_datetime(india_covid_19['Date'],dayfirst = True)

state_testing['Date'] = pd.to_datetime(state_testing['Date'])
train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])
dates = list(confirmed_df.columns[4:])

dates = list(pd.to_datetime(dates))



df1 = confirmed_df.groupby('Country/Region').sum().reset_index()

k = df1[df1['Country/Region']=='Germany'].loc[:,'1/22/20':]

germany_confirmed = k.values.tolist()[0] 

data = pd.DataFrame(columns = ['ds','y'])

data['ds'] = dates

data['y'] = germany_confirmed



prop=Prophet()

prop.fit(data)

future=prop.make_future_dataframe(periods=60)

prop_forecast=prop.predict(future)

# forecast = prop_forecast[['ds','yhat']].tail(60)



fig = plot_plotly(prop, prop_forecast)

fig = prop.plot(prop_forecast,xlabel='Date',ylabel='Confirmed Cases')
arima = ARIMA(data['y'], order=(5, 1, 0))

arima = arima.fit(trend='c', full_output=True, disp=True)

forecast = arima.forecast(steps= 60)

pred = list(forecast[0])



start_date = data['ds'].max()

prediction_dates = []

for i in range(60):

    date = start_date + datetime.timedelta(days=1)

    prediction_dates.append(date)

    start_date = date

plt.figure(figsize= (15,10))

plt.xlabel("Dates",fontsize = 20)

plt.ylabel('Total cases',fontsize = 20)

plt.title("Predicted Values for the next 60 Days" , fontsize = 20)



plt.plot_date(y= pred,x= prediction_dates,linestyle ='dashed',color = '#ff9999',label = 'Predicted');

plt.plot_date(y=data['y'],x=data['ds'],linestyle = '-',color = 'blue',label = 'Actual');

plt.legend();
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

germany_code = countries.index('Germany')

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

test_germany = test[test['Country_Region'] == germany_code]
models = []

mse = []

mae = []

rmse = []
lgbm = LGBMRegressor(n_estimators=1300)

lgbm.fit(x_train,y_train)

pred = lgbm.predict(x_test)

lgbm_forecast = lgbm.predict(test_germany)

models.append('LGBM')

mse.append(round(mean_squared_error(pred, y_test),2))

mae.append(round(mean_absolute_error(pred, y_test),2))

rmse.append(round(np.sqrt(mean_squared_error(pred, y_test)),2))
rf = RandomForestRegressor(n_estimators=100)

rf.fit(x_train,y_train)

pred = rf.predict(x_test)

rfr_forecast = rf.predict(test_germany)

models.append('Random Forest')

mse.append(round(mean_squared_error(pred, y_test),2))

mae.append(round(mean_absolute_error(pred, y_test),2))

rmse.append(round(np.sqrt(mean_squared_error(pred, y_test)),2))
xgb = XGBRegressor(n_estimators=100)

xgb.fit(x_train,y_train)

pred = xgb.predict(x_test)

xgb_forecast = xgb.predict(test_germany)

models.append('XGBoost')

mse.append(round(mean_squared_error(pred, y_test),2))

mae.append(round(mean_absolute_error(pred, y_test),2))

rmse.append(round(np.sqrt(mean_squared_error(pred, y_test)),2))