import numpy as np

import pandas as pd 

import matplotlib.pyplot as mpl

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

from mpl_toolkits.basemap import Basemap

from datetime import datetime

from fbprophet import Prophet

from plotly.subplots import make_subplots



# data1 = pd.read_csv('covid19 (2).csv')

data1 = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')

data1.head()
import copy

data2 = data1.copy()
data2.head()
data2.info()
d = data1['Date'].value_counts().sort_index()
print(d.index[0])

# this is the straing date in dataset
print(d.index[-1])

# this is the ending date in dataset
data1['Active'] = data1['Confirmed']-data1['Deaths']-data1['Recovered']
# VISUALIZATION

mpl.style.use(['ggplot']) 

# for ggplot-like style
data2['Date'] = pd.to_datetime(data2['Date'])

data2['Date'] = data2['Date'].dt.strftime('%m/%d/%Y')

data2 = data2.fillna('-')

fig = px.density_mapbox(data2, lat='Lat', lon='Long', z='Confirmed', radius=20,zoom=1, hover_data=["Country/Region",'Province/State',"Confirmed"],mapbox_style="carto-positron", animation_frame = 'Date', range_color= [0, 2000],title='Spread of Covid-19')

fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

fig.show()
m=Basemap(llcrnrlon=-180, llcrnrlat=-90,urcrnrlon=180,urcrnrlat=90)

# m = Basemap(llcrnrlon=-10.5,llcrnrlat=33,urcrnrlon=10.,urcrnrlat=46., resolution='i', projection='cass', lat_0 = 39.5, lon_0 = 0.)



m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)

m.fillcontinents(color='grey', alpha=0.7, lake_color='grey')

m.drawcoastlines(linewidth=0.1, color="white")

 

# Add a marker per city of the data frame!

m.plot(data1['Lat'], data1['Long'], linestyle='none', marker="o", markersize=4, alpha=0.4, c="yellow",  markeredgewidth=1)
recent = data1[data1['Date'] == data1['Date'].max()]

world = recent.groupby('Country/Region')['Confirmed','Active','Deaths'].sum().reset_index()

world.tail(10)

# len(world)
world['size'] = world['Deaths'].pow(0.2)

fig = px.scatter_geo(world, locations="Country/Region",locationmode='country names', color="Deaths",

                     hover_name="Country/Region", size="size",hover_data = ['Country/Region','Deaths'],

                     projection="natural earth",title='Death count::')

fig.show()
world['size'] = world['Active'].pow(0.2)

fig = px.scatter_geo(world, locations="Country/Region",locationmode='country names', color="Active",

                     hover_name="Country/Region", size="size",hover_data = ['Country/Region','Active'],

                     projection="natural earth",title='active cases in different areas of world::')

fig.show()
top = data1[data1['Date'] == data1['Date'].max()]

recents = top.groupby(by = 'Country/Region')['Confirmed'].sum().sort_values(ascending = False).head(20).reset_index()



mpl.figure(figsize= (10,7))

# mpl.xticks(fontsize = 5)

# mpl.yticks(fontsize = 5)

mpl.xlabel("Total cases")

mpl.ylabel('Country')

mpl.title("Countries with max cases")

ax = sns.barplot(x = recents.Confirmed, y = recents['Country/Region'])

for i, (value, name) in enumerate(zip(recents.Confirmed,recents['Country/Region'])):

    ax.text(value, i-.05, f'{value:,.0f}',  size=10, ha='left',  va='center')

ax.set(xlabel='Total cases', ylabel='Country')
# predicts for future lets say, august



time_series_data = data1[['Date', 'Confirmed']].groupby('Date', as_index = False).sum()

time_series_data.columns = ['ds', 'y']

time_series_data.ds = pd.to_datetime(time_series_data.ds)

time_series_data.head()
train_range = np.random.rand(len(time_series_data)) < 0.8

train_ts = time_series_data[train_range]

test_ts = time_series_data[~train_range]

test_ts = test_ts.set_index('ds')
# now, les try prophet model 
prophet_model = Prophet()

prophet_model.fit(train_ts)
future = pd.DataFrame(test_ts.index)

predict = prophet_model.predict(future)

forecast = predict[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

forecast = forecast.set_index('ds')
prediction_fig = go.Figure() 

prediction_fig.add_trace(go.Scatter(

                x= time_series_data.ds,

                y= time_series_data.y,

                name = "true values",

                line_color= "red",

                opacity= 0.8))

prediction_fig.add_trace(go.Scatter(

                x= forecast.index,

                y= forecast.yhat,

                name = "Predicted values",

                line_color= "yellow",

                opacity= 0.8))

prediction_fig.update_layout(title_text= "Forecasting::", 

                             xaxis_title="time(months)", yaxis_title="prediction of Cases",)



prediction_fig.show()
prophet_model = Prophet()

prophet_model.fit(time_series_data)



future = prophet_model.make_future_dataframe(periods=150)

forecast = prophet_model.predict(future)

forecast = forecast.set_index('ds')



prediction_fig = go.Figure() 

prediction_fig.add_trace(go.Scatter(

                x= time_series_data.ds,

                y= time_series_data.y,

                name = "Actual",

                line_color= "green",

                opacity= 0.8))

prediction_fig.add_trace(go.Scatter(

                x= forecast.index,

                y= forecast.yhat,

                name = "Prediction",

                line_color= "yellow",

                opacity= 0.8))

prediction_fig.update_layout(title_text= "Forecasting", 

                             xaxis_title="time(months)", yaxis_title="prediction of cases",)



prediction_fig.show()