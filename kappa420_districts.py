import numpy as np

import pandas as pd

import folium

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

from sklearn import linear_model

import matplotlib.pyplot as plt

%matplotlib inline

init_notebook_mode(connected = True)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

data = pd.read_csv("../input/crime.csv")

data.head()
data.shape
data.info()
district = data.DISTRICT_ID.value_counts().sort_index()

trace = go.Bar(x = district.index, y = district.values )

layout = go.Layout(title = 'Number of Reported Incidents By Police District',xaxis = dict(title='District ID', ticklen=10))

fig = go.Figure(data = [trace], layout = layout)

iplot(fig)
data['REPORTED_DATE'] = pd.to_datetime(data['REPORTED_DATE'])

DID = data.DISTRICT_ID.unique()

DID.sort()

for x in DID:

    print("District: "+str(x)+"|"+str(min(data[data.DISTRICT_ID == x].REPORTED_DATE)))

DISTBYTIME = data.REPORTED_DATE.dt.date.value_counts().sort_index()

rollmean = DISTBYTIME.rolling(window = 100).mean()

rollstd = DISTBYTIME.rolling(window = 100).std()

trace = go.Scatter(x = DISTBYTIME.index, y = DISTBYTIME.values, mode = 'lines', name = 'Total')

tracemean = go.Scatter(x = DISTBYTIME.index, y=rollmean, mode = 'lines', name = 'Rolling Mean')

tracestd = go.Scatter(x = DISTBYTIME.index, y=rollstd, mode = 'lines', name = 'Rolling Std')

layout = go.Layout(title = 'Number of Reported Incidents vs Time', xaxis = dict(title = 'Date'))

fig = go.Figure(data = [trace, tracemean, tracestd], layout = layout)

iplot(fig)
from sklearn.linear_model import LinearRegression

data['Date'] = data.REPORTED_DATE.dt.date

data['count'] = 1;

total = data.groupby('Date').count()

total['Ticks'] = range(0,len(total.index.values))

lin = LinearRegression()

lin.fit(total[['Ticks']],total[['count']])

countpred = lin.predict(total[['Ticks']])

x = plt.plot(total[['Ticks']],countpred, color = 'black')

plt.xlabel('Days from 01-02-2014')

plt.ylabel('Number of Incidents')

plt.scatter(total[['Ticks']], total[['count']])
trace = []

for x in range(1,8):

    SRC = data.loc[data.DISTRICT_ID == x, 'REPORTED_DATE'].dt.date.value_counts().sort_index()

    trace.append(go.Scatter(x = SRC.index, y = SRC.values, mode = 'lines', name = 'District '+str(x)))

    

layout = go.Layout(title = 'Number of Reported Incidents Separated by District vs Time', xaxis = dict(title = 'Date'))

fig = go.Figure(data = trace, layout = layout)

iplot(fig)
trace = []

rollmean = 0

for x in range(1,7):

    SRC = data.loc[data.DISTRICT_ID == x, 'REPORTED_DATE'].dt.date.value_counts().sort_index()

    rollmean = SRC.rolling(window = 30).mean()

    trace.append(go.Scatter(x = SRC.index, y = rollmean, mode = 'lines', name = 'District '+str(x)))

    

layout = go.Layout(title = 'Rolling Mean of Reported Incidents Separated by District vs Time', xaxis = dict(title = 'Date'))

fig = go.Figure(data = trace, layout = layout)

iplot(fig)
data.OFFENSE_CATEGORY_ID.value_counts()
data = data[~(data.OFFENSE_CATEGORY_ID.isin(['all-other-crimes','traffic-accident','other-crimes-against-persons']))]

data.OFFENSE_CATEGORY_ID.value_counts()
num = 30

num1 = data[data.OFFENSE_CATEGORY_ID =='public-disorder']

crime1 = num1.REPORTED_DATE.dt.date.value_counts().sort_index()

rollmean1 = crime1.rolling(window = num).mean()

num2 = data[data.OFFENSE_CATEGORY_ID =='larceny']

crime2 = num2.REPORTED_DATE.dt.date.value_counts().sort_index()

rollmean2 = crime2.rolling(window = num).mean()

num3 = data[data.OFFENSE_CATEGORY_ID =='theft-from-motor-vehicle']

crime3 = num3.REPORTED_DATE.dt.date.value_counts().sort_index()

rollmean3 = crime3.rolling(window = num).mean()

num4 = data[data.OFFENSE_CATEGORY_ID =='drug-alcohol']

crime4 = num4.REPORTED_DATE.dt.date.value_counts().sort_index()

rollmean4 = crime4.rolling(window = num).mean()

num5 = data[data.OFFENSE_CATEGORY_ID =='auto-theft']

crime5 = num5.REPORTED_DATE.dt.date.value_counts().sort_index()

rollmean5 = crime5.rolling(window = num).mean()

num6 = data[data.OFFENSE_CATEGORY_ID =='burglary']

crime6 = num6.REPORTED_DATE.dt.date.value_counts().sort_index()

rollmean6 = crime6.rolling(window = num).mean()

num2.head()

trace1 = go.Scatter(x = crime1.index, y = crime1.values, mode = 'lines', name = 'Public Disorder')

trace2 = go.Scatter(x = crime2.index, y = crime2.values, mode = 'lines', name = 'Larcency')

trace3 = go.Scatter(x = crime3.index, y = crime3.values, mode = 'lines', name = 'Theft From Motor Vehicles')

trace4 = go.Scatter(x = crime4.index, y = crime4.values, mode = 'lines', name = 'Drug & Alcohol')

trace5 = go.Scatter(x = crime5.index, y = crime5.values, mode = 'lines', name = 'Auto Theft')

trace6 = go.Scatter(x = crime6.index, y = crime6.values, mode = 'lines', name = 'Burglary')

layout = go.Layout(title = 'Top 6 Crimes vs Time')

fig = go.Figure(data = [trace1,trace2,trace3, trace4,trace5, trace6], layout = layout)

iplot(fig)
tracem1 = go.Scatter(x = crime1.index, y = rollmean1, mode = 'lines', name = 'Public Disorder')

tracem2 = go.Scatter(x = crime2.index, y = rollmean2, mode = 'lines', name = 'Larcency')

tracem3 = go.Scatter(x = crime3.index, y = rollmean3, mode = 'lines', name = 'Theft From Motor Vehicles')

tracem4 = go.Scatter(x = crime4.index, y = rollmean4, mode = 'lines', name = 'Drug & Alcohol')

tracem5 = go.Scatter(x = crime5.index, y = rollmean5, mode = 'lines', name = 'Auto Theft')

tracem6 = go.Scatter(x = crime6.index, y = rollmean6, mode = 'lines', name = 'Burglary')

layout = go.Layout(title = 'Top 6 Crimes Rolling Mean(n=30) vs Time')

fig = go.Figure(data = [tracem1,tracem2,tracem3, tracem4,tracem5, tracem6], layout = layout)

iplot(fig)
num = 60

num1 = data[data.OFFENSE_CATEGORY_ID =='public-disorder']

crime1 = num1.REPORTED_DATE.dt.date.value_counts().sort_index()

rollmean1 = crime1.rolling(window = num).mean()

num2 = data[data.OFFENSE_CATEGORY_ID =='larceny']

crime2 = num2.REPORTED_DATE.dt.date.value_counts().sort_index()

rollmean2 = crime2.rolling(window = num).mean()

num3 = data[data.OFFENSE_CATEGORY_ID =='theft-from-motor-vehicle']

crime3 = num3.REPORTED_DATE.dt.date.value_counts().sort_index()

rollmean3 = crime3.rolling(window = num).mean()

num4 = data[data.OFFENSE_CATEGORY_ID =='drug-alcohol']

crime4 = num4.REPORTED_DATE.dt.date.value_counts().sort_index()

rollmean4 = crime4.rolling(window = num).mean()

num5 = data[data.OFFENSE_CATEGORY_ID =='auto-theft']

crime5 = num5.REPORTED_DATE.dt.date.value_counts().sort_index()

rollmean5 = crime5.rolling(window = num).mean()

num6 = data[data.OFFENSE_CATEGORY_ID =='burglary']

crime6 = num6.REPORTED_DATE.dt.date.value_counts().sort_index()

rollmean6 = crime6.rolling(window = num).mean()

num2.head()

trace1 = go.Scatter(x = crime1.index, y = rollmean1, mode = 'lines', name = 'Public Disorder')

trace2 = go.Scatter(x = crime2.index, y = rollmean2, mode = 'lines', name = 'Larcency')

trace3 = go.Scatter(x = crime3.index, y = rollmean3, mode = 'lines', name = 'Theft From Motor Vehicles')

trace4 = go.Scatter(x = crime4.index, y = rollmean4, mode = 'lines', name = 'Drug & Alcohol')

trace5 = go.Scatter(x = crime5.index, y = rollmean5, mode = 'lines', name = 'Auto Theft')

trace6 = go.Scatter(x = crime6.index, y = rollmean6, mode = 'lines', name = 'Burglary')

layout = go.Layout(title = 'Top 6 Crimes Rolling Mean(n=100) vs Time')

fig = go.Figure(data = [trace1,trace2,trace3, trace4,trace5, trace6], layout = layout)

iplot(fig)
year2018 = (data.REPORTED_DATE >= '2018-01-01') & (data.REPORTED_DATE < '2019-01-01')

data = data[year2018]

for i in range(1,len(data.DISTRICT_ID.unique())+1):

    data1 = data[data.DISTRICT_ID == i]

    district1 = data1.OFFENSE_CATEGORY_ID.value_counts()

    labels = district1.index

    values = district1.values

    trace = go.Pie(labels = labels, values = values)

    name = 'District ' + str(i)

    layout = go.Layout(title = name)

    fig = go.Figure(data = [trace], layout = layout)

    iplot(fig)
year2018 = (data.REPORTED_DATE >= '2018-01-01') & (data.REPORTED_DATE < '2019-01-01')

data2018 = data[(year2018) & (data.DISTRICT_ID == 1) & (data.OFFENSE_CATEGORY_ID =='public-disorder')]

idx = data2018['GEO_LAT'].isna() | data2018['GEO_LON'].isna()

data2018 = data2018[~idx]

m = folium.Map(location=[39.76,-105.02], tiles='Stamen Toner',zoom_start=13, control_scale=True)

from folium.plugins import MarkerCluster

mc = MarkerCluster()

for each in data2018.iterrows():

    mc.add_child(folium.Marker(location = [each[1]['GEO_LAT'],each[1]['GEO_LON']]))

m.add_child(mc)

display(m)