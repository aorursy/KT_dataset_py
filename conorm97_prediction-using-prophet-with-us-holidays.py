#import relevant Python libraries

import numpy as np 

import pandas as pd 

from fbprophet import Prophet

import matplotlib.pyplot as plt

%matplotlib inline

import datetime 

from numba import jit

import math

#import required data

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



key = pd.read_csv('../input/key_1.csv')

train = pd.read_csv('../input/train_1.csv')



print(train.shape)
#generate list of dates

dates = list(train.columns.values)[1:]
#convert page views to integers to save memory

for col in train.columns[1:]:

    train[col] = pd.to_numeric(train[col], downcast = 'integer')
#generate 3 psuedo-random between 0 and 145063 (number of articles) using numpy. 

r1 = 19738 

r2 = 7003 

r3 = 41291 
#get randomly chosen time series

r1_ts = train.iloc[r1,:]

r2_ts = train.iloc[r2,:]

r3_ts = train.iloc[r3,:]



print(r1_ts[0])

print(r2_ts[0])

print(r3_ts[0])
#get IDs and page names from the key file

pages = key.Page.values

ids = key.Id.values
#concatenate chosen time series into iterable list 

ts_list = [r1_ts, r2_ts, r3_ts]

days = list(range(550)) 
#plot number of article views against day number for the three time series

for r, n in zip(ts_list, list(range(0,3))):

    plt.plot(days, r[1:])

    plt.ylabel('Number of Views')

    plt.xlabel('Day Number')

    plt.title(r[0])

    plt.show()

   
#drop page column from our time series, not necessary for use with Prophet

for t in ts_list:

    t.drop('Page', inplace=True)

    
#function to create a DataFrame in the format required by Prophet

def create_df(ts):    

    df = pd.DataFrame(columns=['ds','y'])

    df['ds'] = dates

    df = df.set_index('ds')

    df['y'] = ts.values

    df.reset_index(drop=False,inplace=True)

    return df
#get DataFrames suitable for use with Prophet for chosen articles

r1_pro, r2_pro, r3_pro = create_df(r1_ts), create_df(r2_ts), create_df(r3_ts)
#check these are in correct format

print(r1_pro.head())

print(r2_pro.head())

print(r3_pro.head())
#function to remove outliers

def outliers_to_na(ts, devs):

    median= ts['y'].median()

    #print(median)

    std = np.std(ts['y'])

    #print(std)

    for x in range(len(ts)):

        val = ts['y'][x]

        #print(ts['y'][x])

        if (val < median - devs * std or val > median + devs * std):

            ts['y'][x] = None 

    return ts

        

#check number of nan values in each DataFrame prior to outlier removal

print(r1_pro.info())

print('-------------')

print(r2_pro.info())

print('-------------')

print(r3_pro.info())
#remove outliers more than 2 standard deviations from the median number of views during the training period

r1_pro, r2_pro, r3_pro = outliers_to_na(r1_pro, 2), outliers_to_na(r2_pro, 2), outliers_to_na(r3_pro, 2)
print(r1_pro.info())

print('-------------')

print(r2_pro.info())

print('-------------')

print(r3_pro.info())
#dataframe of annual US Public Holidays + 2017 Presidential Inauguration over training and forecasting periods 



ny = pd.DataFrame({'holiday': "New Year's Day", 'ds' : pd.to_datetime(['2016-01-01', '2017-01-01'])})  

mlk = pd.DataFrame({'holiday': 'Birthday of Martin Luther King, Jr.', 'ds' : pd.to_datetime(['2016-01-18', '2017-01-16'])}) 

wash = pd.DataFrame({'holiday': "Washington's Birthday", 'ds' : pd.to_datetime(['2016-02-15', '2017-02-20'])})

mem = pd.DataFrame({'holiday': 'Memorial Day', 'ds' : pd.to_datetime(['2016-05-30', '2017-05-29'])})

ind = pd.DataFrame({'holiday': 'Independence Day', 'ds' : pd.to_datetime(['2015-07-04', '2016-07-04', '2017-07-04'])})

lab = pd.DataFrame({'holiday': 'Labor Day', 'ds' : pd.to_datetime(['2015-09-07', '2016-09-05', '2017-09-04'])})

col = pd.DataFrame({'holiday': 'Columbus Day', 'ds' : pd.to_datetime(['2015-10-12', '2016-10-10', '2017-10-09'])})

vet = pd.DataFrame({'holiday': "Veteran's Day", 'ds' : pd.to_datetime(['2015-11-11', '2016-11-11', '2017-11-11'])})

thanks = pd.DataFrame({'holiday': 'Thanksgiving Day', 'ds' : pd.to_datetime(['2015-11-26', '2016-11-24'])})

christ = pd.DataFrame({'holiday': 'Christmas', 'ds' : pd.to_datetime(['2015-12-25', '2016-12-25'])})

inaug = pd.DataFrame({'holiday': 'Inauguration Day', 'ds' : pd.to_datetime(['2017-01-20'])})



us_public_holidays = pd.concat([ny, mlk, wash, mem, ind, lab, col, vet, thanks, christ, inaug])

#function to calculate in sample SMAPE scores

def smape_fast(y_true, y_pred): #adapted from link to discussion 

    out = 0

    for i in range(y_true.shape[0]):

        if (y_true[i] != None and np.isnan(y_true[i]) ==  False):

            a = y_true[i]

            b = y_pred[i]

            c = a+b

            if c == 0:

                continue

            out += math.fabs(a - b) / c

    out *= (200.0 / y_true.shape[0])

    return out
#function to remove any negative forecasted values.

def remove_negs(ts):

    ts['yhat'] = ts['yhat'].clip_lower(0)

    ts['yhat_lower'] = ts['yhat_lower'].clip_lower(0)

    ts['yhat_upper'] = ts['yhat_upper'].clip_lower(0)
#fit Prophet model and create forecast 

m = Prophet(yearly_seasonality=True, holidays=us_public_holidays)

m.fit(r1_pro)

future = m.make_future_dataframe(periods=31+28, freq='D', include_history=True)

forecast_1 = m.predict(future)

remove_negs(forecast_1)
#plot forecasted values and components 

m.plot(forecast_1)

m.plot_components(forecast_1)
#get in sample SMAPE score

print(smape_fast(r1_pro['y'].values, forecast_1['yhat'].values))
m = Prophet(yearly_seasonality=True, holidays=us_public_holidays)

m.fit(r2_pro)

future = m.make_future_dataframe(periods=31+28, freq='D', include_history=True)

forecast_2 = m.predict(future)
remove_negs(forecast_2)
#plot forecasted values and components 

m.plot(forecast_2)

m.plot_components(forecast_2)
print(smape_fast(r2_pro['y'].values, forecast_2['yhat'].values))
m = Prophet(yearly_seasonality=True, holidays=us_public_holidays)

m.fit(r3_pro)

future = m.make_future_dataframe(periods=31+28, freq='D', include_history=True)

forecast_3 = m.predict(future)
remove_negs(forecast_3)
#plot forecasted values and components 

m.plot(forecast_3)

m.plot_components(forecast_3)
print(smape_fast(r3_pro['y'].values, forecast_3['yhat'].values))