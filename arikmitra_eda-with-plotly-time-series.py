import numpy as np # linear algebra

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df= pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

df_india = df.copy()

df_india.head()
from datetime import datetime

df_india['Date'] = pd.to_datetime(df_india['Date'])

print(df_india['Date'].head())

#print(df_india.head())
#print(df_india.columns)

#df_india.head(10)
#import seaborn as sns

#cm = sns.light_palette("green", as_cmap=True)

#df_india.style.background_gradient(cmap=cm)
effects_df = pd.DataFrame(data=[df_india.Cured,df_india.Deaths,df_india.Confirmed]).transpose()

print(effects_df.head(2))

effects_df.style.background_gradient(cmap='Reds')
#df_india.drop(['Sno'],axis=1,inplace=True)

df_india['Total_Cases'] = df_india['ConfirmedIndianNational'] + df_india['ConfirmedForeignNational']

#print(df_india['Total_Cases'].head())

df_india['Cured/dead'] = df_india['Cured'] + df_india['Deaths']

print(np.dtype(df_india['Cured/dead']))

print(np.dtype(df_india['Total_Cases']))#df_india.drop(['Sno'],axis=1,inplace=True)

df_india['Total_Cases'] = df_india['ConfirmedIndianNational'] + df_india['ConfirmedForeignNational']

#print(df_india['Total_Cases'].head())

df_india['Cured/dead'] = df_india['Cured'] + df_india['Deaths']

print(np.dtype(df_india['Cured/dead']))

print(np.dtype(df_india['Total_Cases']))
df_india['day'] = df_india['Date'].map(lambda x: x.day)

df_india['month'] = df_india['Date'].map(lambda x: x.month)

df_india['year'] = df_india['Date'].map(lambda x: x.year)

print(df_india.head())

print(df_india.columns)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 

import plotly as py
#This is a trial visual

import plotly.express as px



fig = px.bar(df_india, x='State/UnionTerritory', y='Total_Cases', labels={'x':'State/UnionTerritory'},

             color="Confirmed", color_continuous_scale=px.colors.sequential.Brwnyl)

fig.update_layout(title_text='Total COVID-19 cases')

fig.update_yaxes(range=[0,df_india['Total_Cases'].max()])

#fig.show()

py.offline.iplot(fig)
fig = px.bar(df_india, x='day', y='Total_Cases', labels={'x':'State/UnionTerritory'},

             color="State/UnionTerritory", color_continuous_scale=px.colors.sequential.Blugrn_r)

fig.update_layout(title_text='Total COVID-19 cases daily basis statewise')

fig.show()
fig = px.scatter(df_india, x="Total_Cases", y="Date", color="State/UnionTerritory")

fig.show()
df_india.columns
fig = px.scatter_matrix(df_india, dimensions=["Cured/dead","Total_Cases","day"], color="State/UnionTerritory")

fig.show()
import plotly

import plotly.graph_objs as go
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_india['Date'], y=df_india['Total_Cases'], mode='lines+markers',name='Total_Cases'))

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_india['day'], y=df_india['Total_Cases'], mode='lines+markers',name='Total_Cases'))

fig.show()
fig = px.line(df_india, x="day", y="Total_Cases",title='Increase of total cases day wise')

fig.show()
#print(np.sum(df_india['Total_Cases'] == "--"))

#print(np.sum(df_india['Cured/dead'] == "--"))

#print(np.sum(df_india['Cured/dead'] == 0))

#np.int64(df_india['Total_Cases'])

#print(np.nan-0) #the blank spaces can filled with nan since the blank symbol is not a number but a character



#df_india['Total_Cases'] = np.where(df_india['Total_Cases']=="--",np.nan,df_india['Total_Cases'])

#print(np.sum(df_india['Total_Cases'] == "--"))

#print(df_india['Total_Cases'].tail())





#for i in df_india.columns:

#    print(isinstance(i,np.int64))

    

#for i in df_india.columns:

#    print(isinstance(i,str))



#df_india['Total_Active'] = df_india['Total_Cases'] - df_india['Cured/dead']

#print(df_india['Total_Active'].head())

#print(np.unique(df_india['State/UnionTerritory']))
#GLOBAL DATA

#/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv

#/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths_US.csv

#/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv

#/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv

#/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv

#/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv

#/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed_US.csv

#/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv
#DATA FOR INDIA

#/kaggle/input/covid19-corona-virus-india-dataset/patient_wise_data_scrapping_and_cleaning.ipynb

#/kaggle/input/covid19-corona-virus-india-dataset/patients_data.csv

#/kaggle/input/covid19-corona-virus-india-dataset/complete.csv

#/kaggle/input/covid19-corona-virus-india-dataset/web_scraping.ipynb
import pandas as pd

import numpy as np

data1 = pd.read_csv("/kaggle/input/covid19-corona-virus-india-dataset/complete.csv")

#print(data1.head())

print(data1.columns)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 

import plotly as py

import plotly.graph_objs as go

import plotly.express as px
fig = px.line(data1, x="Date", y="Total Confirmed cases",title='Increase of total cases month wise')

fig.show()
data2 = data1.copy()

#print(data2.head())

from datetime import datetime

data2['Date'] = pd.to_datetime(data2['Date'])

april_data = data2.loc[(data2['Date'].dt.month==4)]

print(len(april_data))

print(len(data2)) #making sure we just have data for April in april_data
fig = px.line(april_data, x="Date", y="Total Confirmed cases",title='Increase of total cases for April')

fig.show()
data3 = data1.copy()

data3['Date'] = pd.to_datetime(data3['Date'])

march = data3.loc[data3['Date'].dt.month==3]

print(len(march))

march_april = march.append(april_data,ignore_index=True)

print(march_april.columns)
#len(march_april)

#print(656+496)

#march_april.tail()

fig = px.line(march_april, x="Date", y="Total Confirmed cases",title='Increase of total cases for March & April')

fig.show()

#Drawing a time series plot with a function

#def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):

#    import matplotlib.pyplot as plt

#    plt.figure(figsize=(16,5), dpi=dpi)

#    plt.plot(x, y, color='tab:red')

#    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)

#    plt.show()

    

    

#plot_df(data1,data1['Date'],data1['Total Confirmed cases'],title="Total cases")
import datetime

data4 = data1.copy()

data4['Date'] = pd.to_datetime(data4['Date'])

data4['Month'] = data4['Date'].dt.month

print(data4['Month'].head())

print(data4.columns)
print(len(data4['Month'].unique()))

print(len(data4['Date']))

#label = data4['Month'].unique()
import plotly.express as px

fig = px.box(data4, x="Date", y="Total Confirmed cases")

fig.show()
#Decomposing a time series

#from statsmodels.tsa.seasonal import seasonal_decompose

#from dateutil.parser import parse

#print(data4.columns)