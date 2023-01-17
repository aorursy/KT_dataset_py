# Import libraries

import pandas as pd

pd.options.display.max_columns = 999

import numpy as np

import seaborn as sns

from plotnine import *

import re

import matplotlib

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline
# Import the data set

london = pd.read_csv('../input/london_fire_brigade_service_calls.csv',

    infer_datetime_format=True,

    parse_dates=['timestamp_of_call', 'date_of_call'], )



london['date_of_call'] = london['date_of_call'].dt.date
london.head()
london.info()
london.describe()
#This is an excellent library to get an understanding of each variable in the dataset.

#We will use this to get a quick glimpse and then pick interesting variables for a deeper dive

import pandas_profiling

london_profile = pandas_profiling.ProfileReport(london)
#london_profile.to_file('london_profile.html')
#Feature Creation for response time in minutes

london['First_response_mins']= london['first_pump_arriving_attendance_time']/60

london['Second_response_mins']= london['second_pump_arriving_attendance_time']/60

#Date with year-month format

london['month'] = pd.to_datetime(london['date_of_call']).dt.month
(ggplot(data=london) + aes(x= 'date_of_call', group=1) + stat_count(geom='line') + theme_538() +theme(figure_size=(15,7))

+ scale_x_date(limit=["2017-01-01","2017-04-30"]))
london.groupby(['date_of_call']).size().sort_values(ascending=False)
(ggplot(data=london) + aes(x='hour_of_call',group=1) + geom_bar() + ggtitle(title='Calls by hour of day')

 + theme_538())
plt.figure(figsize=[15,7],)

plt.title('Average First Response Time (mins) by Hour of Call')

sns.swarmplot(data=london,x='hour_of_call', y='First_response_mins')
plt.figure(figsize=[15,7],)

plt.title('Average First Response Time (mins) by Hour of Call')

sns.boxplot(data=london,x='hour_of_call', y='First_response_mins')

plt.plot([-1,24],[6, 6], linewidth=3,linestyle='dashed', )
plt.figure(figsize=[15,7],)

plt.title('Average Second Response Time (mins) by Hour of Call')

sns.boxplot(data=london,x='hour_of_call', y='Second_response_mins')

plt.plot([-1,24],[8, 8], linewidth=3,linestyle='dashed', )
london.groupby(['stop_code_description']).size().sort_values().plot(kind='barh', figsize=[8,4], title='Calls by Incident Type')
plt.figure(figsize=[15,7],)

plt.title('Average First Response Time (mins) by Incident Type')



ranks = london.groupby("stop_code_description")["First_response_mins"].mean().fillna(0).sort_values()[::-1].index

sns.boxplot(data=london, y='stop_code_description', x='First_response_mins', orient='h', order=ranks)
plt.figure(figsize=[15,7],)

plt.title('Average Second Response Time (mins) by Incident Type')



ranks = london.groupby("stop_code_description")["Second_response_mins"].mean().fillna(0).sort_values()[::-1].index

sns.boxplot(data=london, y='stop_code_description', x='Second_response_mins', orient='h', order=ranks)
london[london['stop_code_description'] == 'Special Service'].groupby(

    ['special_service_type']).size().sort_values()
plt.figure(

    figsize=[15, 8], )

plt.title('Average First Response Time (mins) by Special Service Type')



ranks = london[london['stop_code_description'] == 'Special Service'].groupby([

    "special_service_type"

])["First_response_mins"].mean().fillna(0).sort_values()[::-1].index



sns.boxplot(

    data=london[london['stop_code_description'] == 'Special Service'],

    y='special_service_type',

    x='First_response_mins',

    orient='h',

    order=ranks, )
london.groupby(['borough_name']).size().sort_values().plot(kind='barh', figsize= (14,7))
plt.figure(

    figsize=[12, 10], )

plt.title('Average First Response Time (mins) by Borough')



ranks = london.groupby(["borough_name" ])["First_response_mins"].mean().sort_values()[::-1].index

sns.boxplot(

    data=london,

    y='borough_name',

    x='First_response_mins',

    orient='h',

    order=ranks,)
borough_list= london.borough_name.unique()



for i in borough_list:

    print ("")

    print (i)

    print((ggplot(london[london['borough_name']==i]) + 

 aes(x= 'month', group=1) + stat_count(geom='line', ) +theme_538() +theme(figure_size=[2,2])))