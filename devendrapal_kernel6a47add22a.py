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
import matplotlib.pyplot as plt 

import numpy as np

import os 

import pandas as pd 



import plotly.express as px

import datetime

import seaborn as sns

import plotly.graph_objects as go

import warnings

warnings.filterwarnings('ignore')

import folium 

from folium import plugins
df1 = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
df1.head()
covid=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

covid.head()
covid.info()
covid= covid.drop(['SNo'],axis=1)

covid.head()
covid['Province/State'] = covid['Province/State'].fillna('Unknown Location',axis=0)
covid.info()
covid_confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

covid_recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

covid_deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
covid['ObservationDate']= pd.to_datetime(covid['ObservationDate'])

covid['Last Update']=pd.to_datetime(covid['Last Update'])
grouping=covid.groupby('ObservationDate')['Last Update','Confirmed', 'Deaths'].sum().reset_index()
grouping.head()
fig= px.line(grouping,x='ObservationDate' , y='Confirmed',title="Worldwide Confirmed Cases Over Time")

fig.show()
fig = px.line(grouping, x="ObservationDate", y="Deaths", title="Worldwide Deaths Over Time")

fig.show()
df1.head()
data=df1

data['Date']=pd.to_datetime(data.Date,dayfirst=True)
data['confirmed']=data.ConfirmedForeignNational+data.ConfirmedIndianNational
data.head()
data=data.rename(columns={'Date':'date','State/UnionTerritory':'state','Deaths':'deaths'})
latest = data[data['date'] == max(data['date'])].reset_index()

latest.head()
latest_grouped = latest.groupby('state')['confirmed', 'deaths'].sum().reset_index()

latest_grouped.head()
latest = data[data['date'] == max(data['date'])]

latest = latest.groupby('state')['confirmed', 'deaths'].max().reset_index()

latest.head()
latest.sort_values('confirmed')