import pandas as pd 

import numpy as np 

import sys

import warnings

import itertools

from pandas.plotting import autocorrelation_plot

from statsmodels.graphics.tsaplots import plot_acf

from matplotlib import pyplot as plt

from pandas.plotting import lag_plot

import statsmodels.api as sm

import statsmodels.tsa.api as smt

import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

%matplotlib inline

import datetime

import calendar

import seaborn as sns

from statsmodels.tsa.ar_model import AR

from statsmodels.tsa.arima_model import ARMA

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX



warnings.filterwarnings("ignore")
df_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

df_train.fillna('NA',inplace=True)

df_train.head()
df_test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

df_test.fillna('NA',inplace=True)

df_test.head()
df_train['region']=df_train['Province_State']+df_train['Country_Region']

df_train.drop('Province_State',axis=1,inplace=True)

df_train.drop('Country_Region',axis=1,inplace=True)
df_test['region']=df_test['Province_State']+df_test['Country_Region']

df_test.drop('Province_State',axis=1,inplace=True)

df_test.drop('Country_Region',axis=1,inplace=True)
# Define the p, d and q parameters to take any value between 0 and 2

p = range(0,2)

d = range(0,2)

q = range(0,1)
# Generate all different combinations of p, d and q triplets

pdq = list(itertools.product(p, d, q))



# Generate all different combinations of seasonal p, q and q triplets

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
best_aic = np.inf

best_pdq = None

best_seasonal_pdq = None

temp_model = None

for param in pdq:   

    for param_seasonal in seasonal_pdq: 

        temp_model = SARIMAX(df_train.ConfirmedCases,order=param,seasonal_order = param_seasonal,enforce_invertibility=False,

                             enforce_stationarity=False)

        results = temp_model.fit(disp=False)

        if results.aic < best_aic:

            best_aic = results.aic

            best_pdq = param

            best_seasonal_pdq = param_seasonal

print("Best ARIMA {} x {} model - AIC:{}".format(best_pdq,best_seasonal_pdq,best_aic))
countries_list=df_train.region.unique()

distictRegions=[]

for i in countries_list:

    distictRegions.append(df_train[df_train['region']==i])

print("Total Regions =  "+ str(len(countries_list)))



distictTestRegions=[]

for i in countries_list:

    distictTestRegions.append(df_test[df_test['region']==i])
#create the estimates assuming measurement error 

import math

confirmed=[]

fatal=[]

j = 0

for region in distictRegions:

    test_len = len(distictTestRegions[j])

    j = j + 1

    # contrived dataset

    data = region.ConfirmedCases.astype('int32').tolist()

    # fit model

    try:       

        model = SARIMA(data, order=(1,1,0), seasonal_order=(1,1,0,12),measurement_error=True)#seasonal_order=(1, 1, 1, 1))       

        #model = ARIMA(data, order=(3,1,2))

        model_fit = model.fit(disp=False)

        # make prediction

        predicted = model_fit.predict(len(data), len(data)+test_len-1)

        new=np.concatenate((np.array(data),np.array([int(num) for num in predicted])),axis=0)

        confirmed.extend(list(new[-test_len:]))

    except:

        confirmed.extend(list(data[-10:-1]))

        for j in range(34):

            confirmed.append(data[-1]*2)

    

    # contrived dataset

    data = region.Fatalities.astype('int32').tolist()

    # fit model

    try:        

        model = SARIMAX(data, order=(1,1,0), seasonal_order=(1,1,0,12),measurement_error=True)#seasonal_order=(1, 1, 1, 1))

        #model = ARIMA(data, order=(3,1,2))

        model_fit = model.fit(disp=False)

        predicted = model_fit.predict(len(data))

        new=np.concatenate((np.array(data),np.array([int(num) for num in predicted])),axis=0)

        fatal.extend(list(new[-test_len:]))

    except:

        fatal.extend(list(data[-10:-1]))

        for j in range(34):

            fatal.append(data[-1]*2)
df_submit=pd.concat([pd.Series(np.arange(1,1+len(confirmed))),pd.Series(confirmed),pd.Series(fatal)],axis=1)

df_submit=df_submit.fillna(method='pad').astype(int)

df_submit.rename(columns={0: 'ForecastId', 1: 'ConfirmedCases',2: 'Fatalities',}, inplace=True)
df_submit.head() 
df_submit.to_csv('submission.csv',index=False)
