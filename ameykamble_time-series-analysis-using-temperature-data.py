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

import numpy as np

import os

import csv

import math
iot_data = pd.read_csv("../input/iot-data/IOT-temp.csv")

iot_data.head()
print("Count of missing value per column ")

iot_data.isnull().sum()
iot_data["unique_id"] = iot_data.index
print("Geeting detail information of each column")

iot_data.info()
print("Changing name of columns")

iot_data = iot_data.rename(columns = {"room_id/id":"room_id"})

iot_data = iot_data.rename(columns = {"out/in":"out_in"})
print("Maximum temperature in inside and outside is ")

iot_data.groupby("out_in").temp.max()
import datetime 

from datetime import date ,time,datetime
iot_data["date_value"] = iot_data['noted_date'].apply(lambda x: datetime.strptime(x, '%d-%m-%Y %M:%S'))
iot_data["date"] = pd.to_datetime(iot_data.noted_date).dt.date

iot_data["year"] = pd.to_datetime(iot_data.noted_date).dt.year

iot_data["month"] = pd.to_datetime(iot_data.noted_date).dt.month

iot_data["weekday"]= pd.to_datetime(iot_data.noted_date).dt.dayofweek
iot_data.weekday.unique()
condition = [

    (iot_data.weekday ==5),

    (iot_data.weekday ==6),

]



choice = ('weekend','weekend')

iot_data['weekday_tag'] = np.select(condition,choice,default = 'weekday')
iot_data.head()
iot_data1 = iot_data.groupby('month').temp.agg(['min','max'])

iot_data1['month_1'] = iot_data1.index
print("Monthwise minimum and maximum temperature is ")

iot_data1
iot_data8 = pd.merge(left = iot_data,right =iot_data1,how = 'left',left_on = "month" ,right_on = "month_1")

iot_data8.head()
iot_data2 = iot_data1.drop(['month_1'],axis = 1)

iot_data3 = iot_data.groupby(['weekday_tag']).temp.agg(['min','max'])

iot_data4 = iot_data.groupby(['out_in']).temp.agg(['min','max'])

iot_data5 = iot_data.groupby('out_in').unique_id.count()

iot_data6 = iot_data.groupby('weekday_tag').unique_id.count()

iot_data7 = iot_data.groupby('month').unique_id.count()
print("weekday or weekend day wise number of measurement inside and outside temperature")

sns.countplot(x = iot_data.out_in,hue = iot_data.weekday_tag)
print("Minimum and maximum temperature monthwise is ")

iot_data2.plot.bar()
print("Weekwise minimum and maximum temperature")

iot_data3.plot.bar()
print('Minimum and maximu teperature inside and outside room')

iot_data4.plot.bar()
print("Count of meansuremnt inside and outside room")

iot_data5
print("Count of weekday and weekend measurement")

iot_data6
print("Monthwise count of temperature measure")

iot_data7
iot_data9 = iot_data8
import warnings

import itertools

warnings.filterwarnings("ignore")

plt.style.use("fivethirtyeight")

import statsmodels.api as sm

import matplotlib



matplotlib.rcParams['axes.labelsize'] = 14

matplotlib.rcParams['xtick.labelsize'] = 12

matplotlib.rcParams['ytick.labelsize'] = 12

matplotlib.rcParams['text.color'] = 'K'
print("minimum date is ",iot_data9.date.min()," and maximum date is ",iot_data9.date.max())
print("Time series analysis has Started")
print("Deleting unnecessory columns")

iot_data10 =iot_data9.drop(['id','room_id','noted_date','out_in','unique_id','date_value','date','year','weekday','weekday_tag','min','max','month_1'],axis = 1)
print("Grouping data monthwise")

iot_data10 = iot_data10.sort_values('month')

iot_data10.isnull().sum()
print("Finding average temperature monthwise and sorting in accending value also assign date to index")

iot_data10 = iot_data10.groupby(['month']).temp.mean().reset_index()

iot_data10 =  iot_data10.assign(date =['2018-01-01','2018-02-01','2018-03-01','2018-04-01','2018-05-01','2018-06-01','2018-07-01','2018-08-01','2018-09-01','2018-10-01','2018-11-01','2018-12-01'])

iot_data10 = iot_data10.set_index('date')

y = iot_data10.temp

y.head()
print("Average temperature plot monthwise")

y.plot(figsize = (20,6))

plt.show()
p = d = q = range(0,2)

pdq = list(itertools.product(p,d,q))

seasonal_pdq = [(x[0],x[1],x[2],12) for x in list(itertools.product(p,d,q))]

print('Examples of parameter combinations for Seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
print("Selecting best AIC For Model")

for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(y,

                                            order = param,

                                            seasonal_order = param_seasonal,

                                            enforce_stationarity = False,

                                            enforce_invertibility = False)

            result = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,result.aic))

        except :

            continue

print("Selecte AIC is 50")                  
print("Model summary is")

mod = sm.tsa.statespace.SARIMAX(y,

                               order =(1,1,1),

                               param_seasonal = (0,0,0,12),

                               enforce_stationarity = False,

                               enforce_invertibility = False)

result = mod.fit()

print(result.summary().tables[1])
pred = result.get_prediction(start = pd.to_datetime('2018-07-01'),dynamic = False)

pred_ci = pred.conf_int()



ax = y['2018-01-01':'2018-07-01'].plot(label = 'observed')

pred.plot(ax = ax,label ='Forecast',alpha= 0.7,figsize = (100,8))



ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)





ax.set_xlabel('date')

ax.set_ylabel('temp')

plt.legend



plt.show()
pred = result.get_prediction(start = pd.to_datetime('2018-07-01'),dynamic = False)

pred_ci = pred.conf_int()



ax = y['2018-01-01':].plot(label = 'observed')



pred.plot(ax = ax,label ='Forecast',alpha= 0.6,figsize = (14,14))



ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)



ax.set_xlabel('date')

ax.set_ylabel('temp')

plt.legend



plt.show()
print("Predicted and actual temperature table")

y_Forecasted = pred.predicted_mean

y_truth = y['2018-08-01':]

actual_pred_table = y_truth.to_frame().join(y_Forecasted.to_frame())

actual_pred_table = actual_pred_table.rename(columns = {"temp":"Actual_value"})

actual_pred_table = actual_pred_table.rename(columns = {0:"Predicted_value"})

actual_pred_table['error'] = actual_pred_table.apply(lambda x : x.Predicted_value - x.Actual_value,axis = 1)

actual_pred_table['error_square'] = actual_pred_table.apply(lambda x : x.error**2,axis = 1 )

actual_pred_table
print("Error difference between actaul and predicted temperature is ", actual_pred_table.error.mean())

print("mean square error difference between actaul and predicted temperature is ", round(math.sqrt(actual_pred_table.error_square.mean()),0))
pred_uc = result.get_forecast(steps=12)

pred_ci = pred_uc.conf_int()

print("prediction of average temperature for next 12 month is")

pred_ci