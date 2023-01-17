import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn import preprocessing
COVID19_line_list_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv',delimiter=',',encoding='latin-1')

#print('Size of weather data frame is :',data.shape)

COVID19_line_list_data.info()

COVID19_line_list_data[0:10]
COVID19_line_list_data = COVID19_line_list_data[['id','reporting date','location','country','gender','age','summary']]

COVID19_line_list_data.head()
fig = plt.figure(figsize=(24,8))

ax = fig.add_subplot(111)

COVID19_line_list_data.groupby('reporting date').mean().sort_values(by='age', ascending=False)['age'].plot('bar', color='r',width=0.1,title='Cases: reporting, age', fontsize=12)

plt.xticks(rotation = 90)

plt.ylabel('age')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(8)

ax.yaxis.label.set_fontsize(8)

print(COVID19_line_list_data.groupby('reporting date').mean().sort_values(by='age', ascending=False)['age'][[1,2]])

fig = plt.figure(figsize=(24,8))

ax = fig.add_subplot(111)

COVID19_line_list_data.groupby('summary').mean().sort_values(by='age', ascending=False)['age'].plot('bar', color='r',width=0.1,title='Cases: age, summary', fontsize=10)

plt.xticks(rotation = 90)

plt.ylabel('age')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(10)

ax.yaxis.label.set_fontsize(10)

print(COVID19_line_list_data.groupby('summary').mean().sort_values(by='age', ascending=False)['age'][[1,2]])
time_series_covid_19_confirmed = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv',delimiter=',',encoding='latin-1')

#print('Size of weather data frame is :',data.shape)

time_series_covid_19_confirmed.info()

time_series_covid_19_confirmed[0:57]
fig = plt.figure(figsize=(24,8))

ax = fig.add_subplot(111)

time_series_covid_19_confirmed.groupby('Country/Region').mean().sort_values(by='3/14/20', ascending=False)['3/14/20'].plot('bar', color='r',width=0.3,title='time_series_covid_19_confirmed: 3/14/20', fontsize=12)

plt.xticks(rotation = 90)

plt.ylabel('3/14/20')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(12)

ax.yaxis.label.set_fontsize(12)

print(time_series_covid_19_confirmed.groupby('Country/Region').mean().sort_values(by='3/14/20', ascending=False)['3/14/20'][[1,2]])
time_series_covid_19_deaths = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv',delimiter=',',encoding='latin-1')

#print('Size of weather data frame is :',data.shape)

time_series_covid_19_deaths.info()

time_series_covid_19_deaths[0:10]
fig = plt.figure(figsize=(24,8))

ax = fig.add_subplot(111)

time_series_covid_19_deaths.groupby('Country/Region').mean().sort_values(by='3/14/20', ascending=False)['3/14/20'].plot('bar', color='r',width=0.3,title='time_series_covid_19_deaths: 3/14/20', fontsize=12)

plt.xticks(rotation = 90)

plt.ylabel('3/14/20')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(12)

ax.yaxis.label.set_fontsize(12)

print(time_series_covid_19_deaths.groupby('Country/Region').mean().sort_values(by='3/14/20', ascending=False)['3/14/20'][[1,2]])
time_series_covid_19_recovered = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv',delimiter=',',encoding='latin-1')

#print('Size of weather data frame is :',data.shape)

time_series_covid_19_recovered.info()

time_series_covid_19_recovered[0:10]
fig = plt.figure(figsize=(24,8))

ax = fig.add_subplot(111)

time_series_covid_19_recovered.groupby('Country/Region').mean().sort_values(by='3/14/20', ascending=False)['3/14/20'].plot('bar', color='r',width=0.3,title='time_series_covid_19_recovered : 3/14/20', fontsize=12)

plt.xticks(rotation = 90)

plt.ylabel('3/14/20')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(12)

ax.yaxis.label.set_fontsize(12)

print(time_series_covid_19_recovered.groupby('Country/Region').mean().sort_values(by='3/14/20', ascending=False)['3/14/20'][[1,2]])