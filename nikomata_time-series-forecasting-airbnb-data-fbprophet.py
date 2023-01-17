# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import plotly as py

from plotly.offline import iplot, plot, init_notebook_mode, download_plotlyjs

import plotly.graph_objs as go

init_notebook_mode(connected=True)

import plotly.offline as offline



import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#load the availablity per listing_id by date and its price

listing_calendar = pd.read_csv("../input/seattle/calendar.csv")

listing_calendar.head(10)
#print columns information and its data type

listing_calendar.info()
#remove $ sign on the price column and convert to numeric value

listing_calendar['price'] = listing_calendar['price'].apply(lambda x: str(x).replace('$',''))

listing_calendar['price'] = pd.to_numeric(listing_calendar['price'], errors = 'coerce')

df_calendar = listing_calendar.groupby('date')[["price"]].sum()

df_calendar['mean'] = listing_calendar.groupby('date')[["price"]].mean()

df_calendar.columns = ['Total', 'Avg']

df_calendar.head(10)
#SET DATE AS INDEX

df_calendar2 = listing_calendar.set_index("date")

df_calendar2.index = pd.to_datetime(df_calendar2.index)

df_calendar2 = df_calendar2[['price']].resample('M').mean()

df_calendar2.head()
trace3 = go.Scatter(

    x = df_calendar2.index[:-1],

    y = df_calendar2.price[:-1]

)

layout3 = go.Layout(

    title = "Average Prices by Month",

    xaxis = dict(title = 'Month'),

    yaxis = dict(title = 'Price ($)')

)

data3 = [trace3]

figure3 = go.Figure(data = data3, layout = layout3)

offline.iplot(figure3)
import datetime 



#load prophet library

from fbprophet import Prophet



#load 

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



#visualize the data

import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode()



#used for test the accuracy and validation of prophet model

from sklearn.metrics import mean_absolute_error
#have your calendar or event date

#in this tutorial I use US Federal Holidays Date



df_holiday = pd.read_csv('../input/usfederalholidays/us-federal-holidays-2011-2020.csv')

df_holiday['Date'] = pd.to_datetime(df_holiday["Date"]).dt.strftime('%Y-%m-%d')

df_holiday[['ds', 'holiday']] = df_holiday[['Date', 'Holiday']]

df_holiday = df_holiday[['ds', 'holiday']]

df_holiday.head()
#copy the dataframe to your predefined dataframe

#copy only the column that you will use

df_calendar_copy = df_calendar.copy()

df_calendar_copy['date'] = df_calendar_copy.index

df_calendar_copy = df_calendar_copy[['date', 'Avg']]

df_calendar_copy.columns = ['ds', 'y']

df_calendar_copy.head()
#saved original data before it's log-transformed

df_calendar_copy['y_origin'] = df_calendar_copy['y']



#applied log transformation

df_calendar_copy['y'] = np.log(df_calendar_copy['y'])



#convert ds to datetime type

df_calendar_copy['ds'] =  pd.to_datetime(df_calendar_copy['ds'])

df_calendar_copy.head()
#plot the trends

py.iplot([go.Scatter(

    x=df_calendar_copy.ds,

    y=df_calendar_copy.y_origin

)])
#noise detection for input as changepoints date

mean = df_calendar_copy['y'].mean()

stdev = df_calendar_copy['y'].std()



q1 = df_calendar_copy['y'].quantile(0.25)

q3 = df_calendar_copy['y'].quantile(0.75)

iqr = q3 - q1

high = mean + stdev

low = mean - stdev
#define this as changepoints in case you want to filter noise date using mean and standard deviation

df_filtered = df_calendar_copy[(df_calendar_copy['y'] > high) | (df_calendar_copy['y'] < low)]

df_filtered_changepoints = df_filtered



#define this as changepoints in case you want to filter noise date using IQR

filtered_iqr = df_calendar_copy[(df_calendar_copy['y'] < q1 - (1.5 * iqr)) | (df_calendar_copy['y'] < q3 + (1.5 * iqr)) ]
#let's try using mean and standard deviation to get the changepoints

#Create a trace

trace = go.Scatter(

    x = df_calendar_copy['ds'],

    y = df_calendar_copy['y'],

    mode = 'lines',

    name = 'actual data'

)

trace_cp = go.Scatter(

    x = df_filtered_changepoints['ds'],

    y = df_filtered_changepoints['y'],

    mode = 'markers',

    name = 'changepoint'

)



data = [trace,trace_cp]

fig = go.Figure(data=data)

py.offline.iplot(fig)
#instantiate Prophet Object

prophet = Prophet(

                  interval_width = 0.95,

#                   daily_seasonality = True,

                  weekly_seasonality = True,

                  yearly_seasonality = True,

                  changepoint_prior_scale = 0.095, 

#                   changepoints = df_filtered_changepoints['ds'],

                  holidays = df_holiday)



#fit the model to training data , we try to use a whole of data as training data

prophet.fit(df_calendar_copy)



future = prophet.make_future_dataframe(periods = 60, freq = 'd')

future['cap'] = 5.05

forecast = prophet.predict(future)
#make a predicition for 60 datapoints on daily level

#set the cap as defined above on the training data

future = prophet.make_future_dataframe(periods = 60, freq = 'd')

future['cap'] = 5.05

forecast = prophet.predict(future)
#plot the predicted value and observed value

py.iplot([

    go.Scatter(x=df_calendar_copy['ds'], y=df_calendar_copy['y'], name='y'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='yhat'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='upper'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='lower')

])
#merge the actual data and forecasted data to get the error metrics

df_comparison = pd.DataFrame()

df_comparison = pd.merge(df_calendar_copy, forecast, left_on = 'ds', right_on = 'ds')

df_comparison.head()
#print performance metrics

print("MAE yhat\t: {}\nMAE trend\t: {}\nMAE yhat_lower: {}\nMAE yhat_upper: {}".format(

    mean_absolute_error(df_comparison['y'].values,df_comparison['yhat']),

    mean_absolute_error(df_comparison['y'].values,df_comparison['trend']),

    mean_absolute_error(df_comparison['y'].values,df_comparison['yhat_lower']),

    mean_absolute_error(df_comparison['y'].values,df_comparison['yhat_upper'])))
#Print the forecast component

prophet.plot_components(forecast)