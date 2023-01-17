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
# Libraries and environments

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.ticker import PercentFormatter

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')



pd.options.display.max_colwidth = 150
data = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_May19.csv")
data.head()
data.columns
data.shape
for name in data.columns:

    print(name, sum(data[name].isna())*100/2243939)
pd.to_datetime(data['Start_Time']).dt.year.unique()
data['acc_year'] = pd.to_datetime(data['Start_Time']).dt.year

data['acc_month'] = pd.to_datetime(data['Start_Time']).dt.month

data['acc_hr_day'] = pd.to_datetime(data['Start_Time']).dt.hour
data['new_date'] = pd.to_datetime(data['Start_Time']).dt.date
temp = data.groupby('new_date')['ID'].count().reset_index()

fig = go.Figure()

fig.add_trace(go.Scatter(x=temp['new_date'], y=temp['ID']))



fig.update_layout(title_text='Accidents trend over the year',xaxis_rangeslider_visible=True)

fig.show()
data['day_name'] = pd.to_datetime(data['Start_Time']).dt.day_name()
temp = data.groupby(['acc_year', 'day_name'])['ID'].count().reset_index(name = 'count')

fig =go.Figure(go.Sunburst(

    labels=temp['day_name'].values,

    parents=temp['acc_year'].values,

    values=temp['count'].values,

))

fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))



fig.show()
data.groupby('County')['ID'].count().reset_index(name = 'count').sort_values(by = 'count', ascending = False)
temp = data.groupby('County')['ID', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',

       'Visibility(mi)', 'Wind_Speed(mph)',

       'Precipitation(in)'].agg({'ID':'count', 'Temperature(F)':'mean', 'Wind_Chill(F)':'mean', 'Humidity(%)':'mean', 'Pressure(in)':'mean',

       'Visibility(mi)':'mean', 'Wind_Speed(mph)':'mean',

       'Precipitation(in)':'mean'}).reset_index().sort_values(by = 'ID', ascending = False)
temp[temp['ID']>20000].head(20)
temp[temp['ID']<1000].head(20)
data.columns
temp = data.groupby('County')['ID', 'Amenity', 'Bump', 'Crossing',

       'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',

       'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop' ].agg({'ID':'count', 'Amenity':'mean', 'Bump':'mean', 'Crossing':'mean',

       'Give_Way':'mean', 'Junction':'mean', 'No_Exit':'mean', 'Railway':'mean', 'Roundabout':'mean', 'Station':'mean',

       'Stop':'mean', 'Traffic_Calming':'mean', 'Traffic_Signal':'mean', 'Turning_Loop':'mean'}).reset_index().sort_values(by = 'ID', ascending = False)
temp = temp.head(20)

temp
temp.index = temp.County

temp.drop(['County', 'Turning_Loop'], axis = 1, inplace = True)
from sklearn.preprocessing import MinMaxScaler

for name in temp.columns:

    mx = MinMaxScaler()

    temp[name] = mx.fit_transform(temp[name].values.reshape(-1,1))
temp
plt.figure(figsize=(15,7))

plt.pcolor(temp)

plt.yticks(np.arange(0.5, len(temp.index), 1), temp.index)

plt.xticks(np.arange(0.5, len(temp.columns), 1), temp.columns)

plt.show()
data_la = data[data['County']=="Los Angeles"]
data_la.shape
temp = data_la.groupby('new_date')['ID'].count().reset_index()

fig = go.Figure()

fig.add_trace(go.Scatter(x=temp['new_date'], y=temp['ID']))



fig.update_layout(title_text='Accidents trend over the year',xaxis_rangeslider_visible=True)

fig.show()
temp['ID'].mean()
data_la['City'].value_counts()
data_la.columns
data_la[['ID', 'Start_Time', 'Start_Lat', 'Start_Lng',]].to_csv('data_la.csv', index = False)
from IPython.display import YouTubeVideo

YouTubeVideo('1DXyTWV7pes')