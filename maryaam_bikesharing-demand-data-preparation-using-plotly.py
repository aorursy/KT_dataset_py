# load required packages

import pandas as pd

import numpy as np

from scipy import stats

import calendar

from datetime import datetime



#plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import missingno as msno



# Block the warning messages

import warnings 

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))

# load dataset

data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")



data_bike = data_train

# merge train and test sets

#frame = [data_train, data_test]

#data = pd.concat(frame)
data_bike.shape
data_bike.dtypes
data_bike.head(2)
data_bike["date"] = data_bike.datetime.apply(lambda x : x.split()[0])

data_bike["hour"] = data_bike.datetime.apply(lambda x : x.split()[1].split(":")[0])

data_bike["weekDay"] = data_bike.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])

data_bike["month"] = data_bike.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])



# Convert to Category Type

categoryVariableList = ["hour","weekDay","month"]

for var in categoryVariableList:

    data_bike[var] = data_bike[var].astype("category")
data_bike["season"] = data_bike.season.map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })

data_bike["weather"] = data_bike.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\

                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \

                                        3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \

                                        4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })



# Convert to Category Type

categoryVariableList = ["season","weather","holiday","workingday"]

for var in categoryVariableList:

    data_bike[var] = data_bike[var].astype("category")
data_bike.info()
data_bike.describe()
data_bike.isnull().sum()
msno.matrix(data_bike)
data_bike.head()
trace0 = go.Box(y=data_bike["count"], marker=dict(color='#9FA0FF'))

layout = go.Layout(title = 'Boxplot on Count')

fig0 = go.Figure(data=[trace0], layout=layout)

iplot(fig0)
trace1 = go.Box( y=data_bike["count"], x=data_bike["season"], marker=dict(color='#CC7E85'))

layout = go.Layout(title = 'Boxplot on Count across Season')

fig1 = go.Figure(data=[trace1], layout=layout)

iplot(fig1)
trace2 = go.Box( y=data_bike["count"], x=data_bike["hour"], marker=dict(color='#3F8EFC'))

layout = go.Layout(title = 'Boxplot on Count across Hour')

fig2 = go.Figure(data=[trace2], layout=layout)

iplot(fig2)
data_bike_NoOutliers = data_bike[np.abs(data_bike["count"]-data_bike["count"].mean())<=(3*data_bike["count"].std())]
trace4 = go.Box(y=data_bike_NoOutliers["count"], marker=dict(color='#9FA0FF'))

layout = go.Layout(title = 'Boxplot on Count')

fig4 = go.Figure(data=[trace4], layout=layout)

iplot(fig4)
corrMatt = data_bike[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()
trace = go.Heatmap(z = corrMatt, 

                   x = ['temp','atemp','casual','registered','humidity','windspeed','count'], 

                   y = ['temp','atemp','casual','registered','humidity','windspeed','count'])

data=[trace]

iplot(data, filename='basic-heatmap')