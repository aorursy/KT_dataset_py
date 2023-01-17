#Libraries to import

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

import pycountry

import plotly_express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



sns.set_style('darkgrid')

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import OrdinalEncoder

from sklearn import metrics

#import xgboost as xgb

#from xgboost import XGBRegressor

#from xgboost import plot_importance, plot_tree

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import AdaBoostRegressor

import itertools

from sklearn.metrics import mean_squared_error

import statsmodels.api as sm

from math import sqrt
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv') 

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
display(df_train.head())

display(df_train.describe())

display(df_train.info())
df_train['Date'] = pd.to_datetime(df_train['Date'], format = '%Y-%m-%d')

df_test['Date'] = pd.to_datetime(df_test['Date'], format = '%Y-%m-%d')
print('Minimum date from training set: {}'.format(df_train['Date'].min()))

print('Maximum date from training set: {}'.format(df_train['Date'].max()))
print('Minimum date from test set: {}'.format(df_test['Date'].min()))

print('Maximum date from test set: {}'.format(df_test['Date'].max()))
def add_daily_measures(df):

    df.loc[0,'Daily Cases'] = df.loc[0,'ConfirmedCases']

    df.loc[0,'Daily Deaths'] = df.loc[0,'Fatalities']

    for i in range(1,len(df_world)):

        df.loc[i,'Daily Cases'] = df.loc[i,'ConfirmedCases'] - df.loc[i-1,'ConfirmedCases']

        df.loc[i,'Daily Deaths'] = df.loc[i,'Fatalities'] - df.loc[i-1,'Fatalities']

    #Make the first row as 0 because we don't know the previous value

    df.loc[0,'Daily Cases'] = 0

    df.loc[0,'Daily Deaths'] = 0

    return df
df_world = df_train.copy()

df_world = df_world.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()

df_world = add_daily_measures(df_world)
df_world.head()
df_train.Province_State.fillna('NaN', inplace=True)

df_plot = df_train.groupby(['Date','Country_Region','Province_State'], as_index=False)['ConfirmedCases','Fatalities'].sum()
df = df_plot.query("Country_Region=='India'")

df.head()
srcConf = pd.Series(df['ConfirmedCases'].values,

                   index=pd.date_range('2020-01-22',periods=len(df),freq= 'D'))

srcConf.head()
plt.figure(figsize=(12,8))

plt.title('Confirmed cases in India')

plt.xlabel('Days')

plt.ylabel('No. of confirmed cases')

plt.plot(srcConf)

plt.legend(['Confirmed cases'])
# Stationarity test

def stationarity_test(tsObj):

    """Augmented Dickey-Fuller Test for stationarity"""

    from statsmodels.tsa.stattools import adfuller

    print("Results of Dickey-Fuller Test")

    df_test = adfuller(tsObj,autolag='AIC')

    df_out = pd.Series(df_test[0:4],

                      index=['Test Statistic','p-Value','No. of lags used','No. of observations used'])

    print(df_out)
stationarity_test(srcConf)

len(df_train.Country_Region.unique())
p = d = q = range(0, 5)

pdq = list(itertools.product(p, d, q))

print('Examples of parameter combinations for Seasonal ARIMA...')

count=0

for param in pdq:

        count= count+1



print(count)
def find_pqd(srcConf):

    tmp_dic={}

    tmp_rmse=1000000

    tmp_list = []

    breakloop = False

    for param in pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(srcConf,

                                            order=param,

                                            #seasonal_order=param_seasonal,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False,

                                            measurement_error=True)

            results = mod.fit()

            rmse = sqrt(mean_squared_error(srcConf.values, results.fittedvalues))

            if(rmse<tmp_rmse):

                tmp_rmse=rmse

                tmp_dic.update({rmse:param})

#                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic)) 

        except:

            continue

    return tmp_dic.get(tmp_rmse)
tmp_df = df_train[df_train.Country_Region=='India']

ts = pd.Series(tmp_df['ConfirmedCases'].values,

                           index=pd.date_range(tmp_df.Date.min(),

                            periods=len(tmp_df),

                            freq= 'D'))

best_pdq_conf = find_pqd(ts)

best_pdq_conf
tp = sm.tsa.statespace.SARIMAX(srcConf,

                                            order=(4, 0, 3),

                                            #seasonal_order=param_seasonal,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False,

                                            measurement_error=True).fit(disp=False)

sqrt(mean_squared_error(srcConf.values, tp.fittedvalues))
plt.figure(figsize=(12,8))

plt.title('Confirmed cases in India')

plt.xlabel('Days')

plt.ylabel('No. of confirmed cases')

plt.plot(srcConf)

plt.plot(tp.fittedvalues)

plt.legend(['Confirmed cases','Fitted'])
df_train.head()
df_train.Province_State.fillna('NaN', inplace=True)

df_test.Province_State.fillna('NaN', inplace=True)
submission = []

#Loop through all the unique countries

for country in df_train.Country_Region.unique():

    #Filter on the basis of country

    df_train1 = df_train[df_train["Country_Region"]==country]

    #Loop through all the States of the selected country

    for state in df_train1.Province_State.unique():

        #Filter on the basis of state

        df_train2 = df_train1[df_train1["Province_State"]==state]

        #Timeseries dataframe

        df_train3_conf = pd.Series(df_train2['ConfirmedCases'].values,

                           index=pd.date_range(df_train2.Date.min(),

                            periods=len(df_train2),

                            freq= 'D'))

        df_train3_fat = pd.Series(df_train2['Fatalities'].values,

                           index=pd.date_range(df_train2.Date.min(),

                            periods=len(df_train2),

                            freq= 'D'))

        best_pdq_conf = find_pqd(df_train3_conf)

        best_pdq_fat = find_pqd(df_train3_fat)

        #model for predicting Confirmed cases

        model1 = sm.tsa.statespace.SARIMAX(df_train3_conf,

                                order=best_pdq_conf,

                                #seasonal_order=best_pdq_conf[1],

                                enforce_stationarity=False,

                                enforce_invertibility=False,

                                measurement_error=True)

        conf = model1.fit(disp=False)

        #model2 for predicting Fatalities

        model2 = sm.tsa.statespace.SARIMAX(df_train3_fat,

                                order=best_pdq_fat,

                                #seasonal_order=best_pdq_fat[1],

                                enforce_stationarity=False,

                                enforce_invertibility=False,

                                measurement_error=True)

        fat = model2.fit(disp=False)

        #Get the test data for that particular country and state

        df_test1 = df_test[(df_test["Country_Region"]==country) & (df_test["Province_State"] == state)]

        #Store the ForecastId separately

        ForecastId = df_test1.ForecastId.values

        conf_pred = conf.get_prediction(start= df_test1.Date.min(), end= df_test1.Date.max(),dynamic=False).predicted_mean

        conf_pred = [round(p) if p>0 else 0 for p in conf_pred]

        fat_pred = fat.get_prediction(start= df_test1.Date.min(), end= df_test1.Date.max(),dynamic=False).predicted_mean

        fat_pred = [round(p) if p>0 else 0 for p in fat_pred]

        #Append the predicted values to submission list

        for i in range(len(conf_pred)):

            d = {'ForecastId':ForecastId[i], 'ConfirmedCases':conf_pred[i], 'Fatalities':fat_pred[i]}

            submission.append(d)
submission
df_submit = pd.DataFrame(submission)
df_submit.head()
len(df_test)
df_submit.to_csv('submission.csv', index=False)