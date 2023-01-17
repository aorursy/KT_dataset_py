# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
airstaday=pd.read_csv("/kaggle/input/air-quality-data-in-india/station_day.csv")

airstaday.head()
airstat=pd.read_csv("/kaggle/input/air-quality-data-in-india/stations.csv")

airstat
aircity=pd.read_csv("/kaggle/input/air-quality-data-in-india/city_day.csv")

aircity
from datetime import datetime

for i in range(len(aircity)):

   aircity.loc[i,'Year'] = datetime.strptime(aircity.loc[i,'Date'], '%Y-%m-%d').year

aircity.Year=aircity.Year.astype("int64")

aircitypivot=pd.pivot_table(aircity,['AQI','CO', 'PM2.5','PM10','NO2','SO2','O3'], 'Year') 

aircitypivot=aircitypivot.reset_index('Year') 

aircitypivot
import plotly.express as px



fig = px.line(aircitypivot, x="Year", y="AQI", title='Avg AQI')

fig.show()
import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Scatter(x=aircitypivot['Year'], y=aircitypivot['PM10'],

                    mode='lines',name='PM10'))

fig.add_trace(go.Scatter(x=aircitypivot['Year'], y=aircitypivot['PM2.5'],

                    mode='lines+markers', name='PM2.5'))





fig.add_trace(go.Scatter(x=aircitypivot['Year'], y=aircitypivot['O3'], name='O3',

                         line=dict(color='green', width=4,dash='dashdot'))) 

fig.add_trace(go.Scatter(x=aircitypivot['Year'], y=aircitypivot['NO2'], name='NO2',

                         line=dict(color='firebrick', width=4,dash='dash'))) 



fig.add_trace(go.Scatter(x=aircitypivot['Year'], y=aircitypivot['SO2'], name='SO2',

                         line=dict(color='royalblue', width=4,dash='dot'))) 

fig.add_trace(go.Scatter(x=aircitypivot['Year'], y=aircitypivot['CO'],

                    mode='lines+markers',name='CO'))



fig.update_layout(title='Average Air Pollutant Levels', xaxis_title='Year' )



fig.show()
aircitypivot1=pd.pivot_table(aircity,['AQI'], 'Year','City') 

#aircitypivot1=aircitypivot1.reset_index('Year') 

aircitypivot1
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [19.5, 16]

aircitypivot1.plot(kind='bar', title="Avg AQI for States")

plt.show()

import seaborn as sns

g = sns.FacetGrid(aircity1, row='Year', col='AQI_Bucket', height=4)

g.map(plt.hist, 'City', alpha=0.5, bins=15)

g.add_legend()

plt.show()


aircity20=aircity.loc[aircity['Year'] == 2020]

aircity20=aircity20.drop(["Year",'NO','NO2','PM2.5','PM10','CO','SO2','O3','NH3','NOx','Benzene','Toluene','Xylene'], axis=1)

aircity20=aircity20.dropna()
cm = sns.light_palette("orange", as_cmap=True)

aircity20.style.background_gradient(cmap=cm)



fig = px.bar(aircity20, x='AQI_Bucket', y='City', title="AQI Category Level for 2020")

fig.show()
plt.rcParams['figure.figsize'] = [19.5, 16]

#sns.relplot(x="Date", y="AQI_Bucket", data=aircity20)

plt.scatter(x=aircity20['AQI_Bucket'], y=aircity20['Date'])