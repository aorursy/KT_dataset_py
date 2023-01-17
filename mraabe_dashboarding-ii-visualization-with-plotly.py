# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#import dataset
df_callcenter = pd.read_csv ('../input/service-requests-received-by-the-oakland-call-center.csv')
#df_callcenter.sample(3)
# first visualization: Histogram with count of council districts of the requests
plt.figure(figsize=(15,8))
first_diagram = sns.countplot(x ="COUNCILDISTRICT",  palette="rocket", data = df_callcenter).set_title('Number of requests per Council District')

#second visualization: count of status
plt.figure(figsize=(15,8))
second_diagram = sns.countplot(y= 'STATUS',palette="rocket", data=df_callcenter).set_title('Request status')
'''Fill in the Null / NaN values: DATETIMEINIT has no NaN values; 
DATETIMECLOSED is filled with the current business date
--> pd.to_datetime('now')
''' 

data = {'REQUESTID': df_callcenter['REQUESTID'],
        'REQCATEGORY': df_callcenter['REQCATEGORY'],
    'DATETIMEINIT': df_callcenter['DATETIMEINIT'],
 'DATETIMECLOSED': df_callcenter['DATETIMECLOSED'].fillna(pd.to_datetime('now'))}

df_callcenter_diff = pd.DataFrame(data, columns = ('REQUESTID','REQCATEGORY','DATETIMEINIT','DATETIMECLOSED'))

# drop rows where a value is null: only REQCATEGORY is null
df_callcenter_diff = df_callcenter_diff.dropna()
#df_callcenter_diff.sample(3)
from datetime import datetime
df_callcenter_diff['DATETIMEINIT_new'] = pd.to_datetime(df_callcenter_diff['DATETIMEINIT']).astype('datetime64[D]')
df_callcenter_diff['DATETIMECLOSED_new'] = pd.to_datetime(df_callcenter_diff['DATETIMECLOSED']).astype('datetime64[D]')
df_callcenter_diff['TIME_DIFF'] = df_callcenter_diff['DATETIMECLOSED_new'] - df_callcenter_diff['DATETIMEINIT_new']
df_callcenter_diff['TIME_DIFF_days'] = df_callcenter_diff['TIME_DIFF'].dt.days
#df_callcenter_diff.sample(3)
grouped = df_callcenter_diff.groupby('REQCATEGORY')
grouped_mean = grouped['TIME_DIFF_days'].agg(np.mean)

plt.figure(figsize=(15,8))
ax = grouped_mean.plot(kind = 'bar', color = 'r').set_title('Average time (days) to close request per request category')
df_district = df_callcenter_diff
df_district['COUNCILDISTRICT'] = df_callcenter['COUNCILDISTRICT']
#df_district.sample(5)

# number of requests per COUNCILDISTRICT
# get location information for Oakland? 
request_per_district = df_district['COUNCILDISTRICT'].value_counts()
request_per_district = request_per_district.to_frame()
request_per_district.index.names = ['COUNCILDISTRICT']
request_per_district.columns = ['NO_REQUESTS']
#request_per_district.head()


# initiated requests per year
import datetime as dt

request_per_year = df_callcenter_diff
request_per_year['YEAR'] = pd.to_datetime(df_callcenter_diff['DATETIMEINIT']).dt.year
#request_per_year.head()

plot_data = request_per_year.groupby('YEAR')['REQUESTID'].count().to_frame()
#plot_data = request_per_year['YEAR'].value_counts()
plot_data = plot_data.reset_index(drop = False)
#plot_data.columns = ['YEAR', 'REQUESTID']
#plot_data


# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Scatter(x=plot_data['YEAR'], y=plot_data['REQUESTID'])]

# specify the layout of our figure
layout = dict(title = "Total number of requests per year",
              xaxis= dict(title= 'Year',ticklen= 10,zeroline= True))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)
open_requests = df_callcenter.loc[df_callcenter['STATUS'].isin(['OPEN','PENDING'])]
open_requests = open_requests.groupby('REQCATEGORY')['REQUESTID'].count().to_frame()
open_requests.reset_index(drop = False, inplace = True)
open_requests.columns

#plot the data
import plotly.plotly as py
import plotly.graph_objs as go

data = [go.Bar(
            x=open_requests['REQCATEGORY'],
            y=open_requests['REQUESTID']
    )]

layout = dict(title = "Open requests per category",
              xaxis= dict(title= '',ticklen= 5,zeroline= True),
              yaxis= dict(title= 'count',ticklen= 5,zeroline= True))

fig = dict(data = data, layout = layout)
iplot(fig)
