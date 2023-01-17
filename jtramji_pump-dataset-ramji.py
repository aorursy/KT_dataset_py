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
#importing the data set
import pandas as pd
sensor = pd.read_csv("../input/pump-sensor-data/sensor.csv",index_col='timestamp')
sensor.head()
#dropping unnamed column
sensor = sensor.drop('Unnamed: 0',axis=1)
#Going to do some EDA for Pump Data
#first we will makle time stamp column to Time 
#pd.to_datetime(sensor['timestamp'])
#examining the dataset with head and tail

sensor.head()
sensor.tail()
sensor.head()
#For observing the data information there are 53 columns as 52 as predictor columns and one as Target column
#Observing the data we can find sensor 15 doesn't have any value so we can drop it.
sensor.info()
sensor= sensor.drop('sensor_15',axis=1)
#these are the dataset with the left columns
sensor.info()
#finding the missing values and replacing it 

sensor.isnull().sum()
#As we find more null values in sensor00,sensor50,sensor_51 we are removing 3 columns .though it is not a good practice to 
#columns i am leaving it for now and lets see later if non dropping makes sense

sensor = sensor.drop(['sensor_00','sensor_50','sensor_51'],axis=1)
#checking null values and filling it
sensor.isnull().sum()
sensor=sensor.ffill(axis=0)
#checking the null values again, Now there are no null values in the dataset 
sensor.isnull().sum()
#seeing the final columns in the dataset 48 independent col and 1 target col
sensor.columns
sensor['machine_status'].unique()
#converting the categorical values into numerical ones
sensor['machine_status'] = sensor['machine_status'].apply(lambda x : 1 if x =='NORMAL'else (0 if x =='BROKEN' else 2))
sensor['machine_status'].unique()
sensor.head(5)
sensor.head(1)
sensor.tail(1)
sensor.index[0],sensor.index[-1]
# lets do some data analysis 
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

sns.distplot(sensor['sensor_01'])
sns.distplot(sensor['sensor_02'])
sns.distplot(sensor['sensor_03'])
sns.distplot(sensor['sensor_04'])
sns.distplot(sensor['sensor_05'])
sns.distplot(sensor['sensor_06'])
sns.distplot(sensor['sensor_07'])
sns.distplot(sensor['sensor_08'])
plt.show()

plt.figure(figsize=(24,20))
sensor.plot()
plt.show()

import seaborn as sns
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import json
import numpy as np
sensor.columns
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=10, cols=1)

fig.append_trace(
    go.Scatter(x=sensor.index, y=sensor['sensor_01']),
    row=1, col=1
)

fig.append_trace(
    go.Scatter(x=sensor.index, y=sensor['sensor_02']),
    row=2, col=1
)

fig.append_trace(
    go.Scatter(x=sensor.index, y=sensor['sensor_03']),
    row=3, col=1
)

fig.append_trace(
    go.Scatter(x=sensor.index, y=sensor['sensor_04']),
    row=4, col=1
)

fig.append_trace(
    go.Scatter(x=sensor.index, y=sensor['sensor_05']),
    row=5, col=1
)

fig.append_trace(
    go.Scatter(x=sensor.index, y=sensor['sensor_06']),
    row=6, col=1
)

fig.append_trace(
    go.Scatter(x=sensor.index, y=sensor['sensor_07']),
    row=7, col=1
)

fig.append_trace(
    go.Scatter(x=sensor.index, y=sensor['sensor_08']),
    row=8, col=1
)

fig.append_trace(
    go.Scatter(x=sensor.index, y=sensor['sensor_09']),
    row=9, col=1
)

fig.append_trace(
    go.Scatter(x=sensor.index, y=sensor['sensor_10']),
    row=10, col=1
)

fig.update_layout(height=600, width=900, title_text="stacked_plots",xaxis = dict(title = 'minuteby minutedata'),
                yaxis = dict(title = 'Sensor values'))
fig.show()



