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




import pandas as pd

import numpy as np

import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt

%matplotlib inline
# Reading the dataset

data= pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

data.head()
data.describe()
# converting the date format for processing and visualisations

data['Date'] = pd.to_datetime(data['ObservationDate'])
def china_row(col):

    Country=col[0]

    if Country == "Mainland China":

        return "China"

    else:

        return "ROW"
data['China_ROW'] = data[['Country/Region']].apply(china_row,axis=1)
data.head()
data_latest = pd.DataFrame(data.groupby(['Country/Region','Date'])['Confirmed','Recovered','Deaths'].sum()).reset_index()

data_latest = data_latest.sort_values(by=['Country/Region','Date'])

data_latest = data_latest.drop_duplicates(subset = ['Country/Region'],keep='last')

data_latest.head()
fig = px.choropleth(data_latest,locations="Country/Region",locationmode='country names',color='Confirmed'

                   ,hover_data=['Confirmed','Recovered','Deaths'],color_continuous_scale="viridis",

                   title='Confirmed cases across the globe')

fig.show()

data_china = data[data['China_ROW'] == "China"]
data_china = pd.DataFrame(data_china.groupby(['Province/State','Date'])['Confirmed','Recovered','Deaths'].sum()).reset_index()
data_china = data_china.sort_values(by=['Province/State','Date'])

data_china_latest = data_china.drop_duplicates(subset = ['Province/State'],keep='last')

data_china['Recover_rate'] = (data_china.Recovered / data_china.Confirmed) *100

df_china = pd.melt(data_china_latest[['Province/State','Confirmed','Recovered','Deaths']],id_vars='Province/State',var_name='Status',value_name='Count')
fig = plt.figure(figsize=(16,8))

ax = sns.barplot(x='Province/State',y='Count',hue='Status',data=df_china)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax.set_title("Current Status (Mainland China)")
df_hubei = data_china[data_china['Province/State'] == 'Hubei']

df_hubei.head()
fig, ax = plt.subplots(figsize=(18,8))

sns.lineplot(data=df_hubei,x='Date',y='Confirmed',marker='o',ax=ax,color='Orange')

sns.lineplot(data=df_hubei,x='Date',y='Recovered',ax=ax,color='g')

sns.lineplot(data=df_hubei,x='Date',y='Deaths',marker='o',ax=ax,color='red')

ax.legend(['Confimed','Recovered','Deaths'])

ax.set(xlabel='Date',ylabel='Total Count')

ax.set_title("Daily Trend for Hubei Province")

plt.show()
fig, ax = plt.subplots(figsize=(16,8))

sns.lineplot(data=data_china,x='Date',y='Recover_rate',hue='Province/State',marker='o',ax=ax)

ax.set(xlabel='Date',ylabel='Total Count')

ax.set_title("Recovery rate of each province within Mainland China")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
data_row = data[data['China_ROW'] == "ROW"]

data_row = pd.DataFrame(data_row.groupby(['Country/Region','Date'])['Confirmed','Recovered','Deaths'].sum()).reset_index()

data_row = data_row.sort_values(by=['Country/Region','Date'])

data_row_latest = data_row.drop_duplicates(subset = ['Country/Region'],keep='last')



df_row_top = data_row_latest[data_row_latest['Confirmed'] > 49]

df_row = pd.melt(df_row_top[['Country/Region','Confirmed','Recovered','Deaths']],id_vars='Country/Region',var_name='Status',value_name='Count')

df_row.shape
fig = px.choropleth(data_row_latest,locations="Country/Region",locationmode='country names',color='Confirmed'

                   ,hover_data=['Confirmed','Recovered','Deaths'],color_continuous_scale="viridis",

                   title='Confirmed cases across Rest of the world')

fig.show()
fig = plt.figure(figsize=(16,8))

ax = sns.barplot(x='Country/Region',y='Count',hue='Status',data=df_row)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax.set_title("Current Status Rest of the World (50 or more confirmed)")
data_china_daily = pd.DataFrame(data_china.groupby(data_china['Date'])['Confirmed','Recovered','Deaths'].sum()).reset_index()

data_china_daily.head()
data_row.head()

data_row_daily = pd.DataFrame(data_row.groupby(data_row['Date'])['Confirmed','Recovered','Deaths'].sum()).reset_index()



fig, ax = plt.subplots(figsize=(16,8))

sns.lineplot(data=data_china_daily,x='Date',y='Confirmed',marker='o',ax=ax,color='Orange')

sns.lineplot(data=data_china_daily,x='Date',y='Recovered',marker='o',ax=ax,color='g')

sns.lineplot(data=data_row_daily,x='Date',y='Confirmed',ax=ax,color='r',marker='o')

sns.lineplot(data=data_row_daily,x='Date',y='Recovered',ax=ax,color='b',marker='o')

#sns.lineplot(data=df_hubei,x='day',y='Deaths',marker='o',ax=ax,color='red')

ax.legend(['Confimed [China]','Recovered [China]','Confimed [Rest of world]','Recovered [Rest of world]'])

ax.set(xlabel='Date',ylabel='Total Count')

ax.set_title("Daily Trend for Confirmed and Recovered cases (China vs Rest of world)")

plt.show()
data_row_daily['Mortality'] = data_row_daily.Deaths / data_row_daily.Confirmed

data_china_daily['Mortality'] = data_china_daily.Deaths / data_china_daily.Confirmed



fig, ax = plt.subplots(figsize=(16,8))

sns.lineplot(data=data_row_daily,x='Date',y='Mortality',marker='o',ax=ax,color='Orange')

sns.lineplot(data=data_china_daily,x='Date',y='Mortality',ax=ax,color='g')



ax.legend(['Mortality rate (Rest of world)','Mortality rate (China)'])

ax.set(xlabel='Date',ylabel='Mortality Rate')

#ax.set(xticks=data_china_daily.day.values)



ax.set_title("Mortality Rate China vs Rest of World")

plt.show()
fig = px.treemap(data_row_latest,path=['Country/Region'],values='Deaths',

                title="Latest number of deaths outside china",

                color_discrete_sequence=px.colors.qualitative.Prism)

fig.show()