import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/train (3).csv')
df.shape
# see if there any null values

df.isnull().sum()
data = pd.DataFrame(df)
data
data.info()

## we have only 2 numerical columns (ConfirmedCases,Fatalities)
## we will change (Date) column datatype from object to DateTime datatype

data['Date'] = pd.to_datetime(data['Date'])

data['Date'].astype
# we will classify Date column to (Day , month ) to make some plots

# we won't make one for year because it is only one year 2020 so it will not be useful

data['Date_month']=data['Date'].dt.month

data['Date_day']=data['Date'].dt.day
plt.figure(figsize=(10,10))

data['Country_Region'].value_counts().head(50).plot(kind='barh')
# make the countplot for Date_month

sns.countplot(data['Date_month'])
# make the countplot for Date_day

sns.countplot(data['Date_day'])
## i will remove id column as it is not useful right now

data = data.drop(columns='Id')
sns.pairplot(data)
plt.figure(figsize=(15,8))

sns.heatmap(data.corr(),annot=True)
##  scatterplot using Seaborn

sns.scatterplot(x='ConfirmedCases',y='Fatalities',data = data)
##  scatter using matplotlib

plt.figure(figsize=(10,6))

plt.scatter(x='ConfirmedCases',y='Fatalities',data = data)
# if we try to make countplot on (ConfirmedCases)column 

# it will not give us a reasonalble visualization because we have alot of unique values but i will give you simple example  

# for the first 50 rows to extract the uniqe values in it

Cc_ex = (data['ConfirmedCases']).iloc[:50]

sns.countplot(Cc_ex)
# boxplot help us in data analysis useful for indicating whether a distribution is skewed and whether 

#there are potential unusual observations (Outliers)

sns.boxplot(data['ConfirmedCases'])
# if we want to know the null values using heatmap

plt.figure(figsize=(10,6))

sns.heatmap(data.isnull(), cbar=False)
plt.figure(figsize=(15,4))

plt.plot(data['Date_month'].value_counts().sort_index())
import plotly

import plotly.graph_objects as go

import cufflinks as cf

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected= True)

cf.go_offline()
trace_a = go.Bar(x=data['Date'],

                y=data['ConfirmedCases'],

                name='Fatalities',

                marker=dict(color='#A2D5F2'))





data3 = go.Data([trace_a])



#data3 = [go.Bar(x=df_inaug.Year, y=df_inaug.America)]



plotly.offline.iplot(data3, filename='jupyter/basic_bar')

trace_a = go.Bar(x=data['Country_Region'],

                y=data['Fatalities'],

                name='Fatalities',

                marker=dict(color='red'))





data3 = go.Data([trace_a])



#data3 = [go.Bar(x=df_inaug.Year, y=df_inaug.America)]



mylayout = go.Layout(

    title="Frequency of Covid-19 Fatalities respect to the Country Region"

)



fig = go.Figure(data=data3, layout=mylayout)

plotly.offline.iplot(fig, filename='jupyter/basic_bar')
trace_a = go.Bar(x=data['Province_State'],

                y=data['Fatalities'],

                name='Fatalities',

                marker=dict(color='red'))





data3 = go.Data([trace_a])



#data3 = [go.Bar(x=df_inaug.Year, y=df_inaug.America)]



mylayout = go.Layout(

    title="Frequency of Covid-19 Fatalities respect to the Province State"

)



fig = go.Figure(data=data3, layout=mylayout)

plotly.offline.iplot(fig, filename='jupyter/basic_bar')
# Histogram for the country region

import plotly.express as px

fig = px.histogram(data, x="Country_Region")

fig.show()
# Histogram for the date by month

import plotly.express as px

fig = px.histogram(data, x="Date_month")

fig.show()
# Histogram for the date by day

import plotly.express as px

fig = px.histogram(data, x="Date_day")

fig.show()
data.iplot(kind='bar',x='Date_month')
data.Country_Region.iplot()
data.Province_State.iplot()
data['Country_Region'].iplot(kind='barh')