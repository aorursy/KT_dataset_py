import numpy as np 
import pandas as pd
import warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
from scipy.special import boxcox
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#PLOTLY
import plotly
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
from plotly.graph_objs import Scatter, Figure, Layout
cf.set_config_file(offline=True)
train = pd.read_csv("../input/train.csv", nrows = 30_000)
test = pd.read_csv("../input/test.csv",nrows = 30_000)
print(">>  Data Loaded")
train.head()
test.head()
print(train.dtypes)
train['key'] = pd.to_datetime(train['key'])
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
print(train.dtypes)
print(f"Numer of Missing values in train: ", train.isnull().sum().sum())
print(f"Number of Missing values in test: ", test.isnull().sum().sum())
print("Train shape {}".format(train.shape))
print("Test shape {}".format(test.shape))
target = train.fare_amount
data = [go.Histogram(x=target)]
layout = go.Layout(title = "Fare Amount Histogram")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
train.head()
print(f">> Data Available since {train.key.min()}")
print(f">> Data Available upto {train.key.max()}")
train.head()
data = [go.Scattermapbox(
            lat= train['pickup_latitude'] ,
            lon= train['pickup_longitude'],
            customdata = train['key'],
            mode='markers',
            marker=dict(
                size= 4,
                color = 'gold',
                opacity = .8,
            ),
          )]
layout = go.Layout(autosize=False,
                   mapbox= dict(accesstoken="pk.eyJ1Ijoic2hhejEzIiwiYSI6ImNqYXA3NjhmeDR4d3Iyd2w5M2phM3E2djQifQ.yyxsAzT94VGYYEEOhxy87w",
                                bearing=10,
                                pitch=60,
                                zoom=13,
                                center= dict(
                                         lat=40.721319,
                                         lon=-73.987130),
                                style= "mapbox://styles/shaz13/cjiog1iqa1vkd2soeu5eocy4i"),
                    width=900,
                    height=600, title = "Pick up Locations in NewYork")
fig = dict(data=data, layout=layout)
iplot(fig)
data = [go.Scattermapbox(
            lat= train['dropoff_latitude'] ,
            lon= train['dropoff_longitude'],
            customdata = train['key'],
            mode='markers',
            marker=dict(
                size= 4,
                color = 'cyan',
                opacity = .8,
            ),
          )]
layout = go.Layout(autosize=False,
                   mapbox= dict(accesstoken="pk.eyJ1Ijoic2hhejEzIiwiYSI6ImNqYXA3NjhmeDR4d3Iyd2w5M2phM3E2djQifQ.yyxsAzT94VGYYEEOhxy87w",
                                bearing=10,
                                pitch=60,
                                zoom=13,
                                center= dict(
                                         lat=40.721319,
                                         lon=-73.987130),
                                style= "mapbox://styles/shaz13/cjk4wlc1s02bm2smsqd7qtjhs"),
                    width=900,
                    height=600, title = "Drop off locations in Newyork")
fig = dict(data=data, layout=layout)
iplot(fig)
train.head(2)
train['pickup_datetime_month'] = train['pickup_datetime'].dt.month
train['pickup_datetime_year'] = train['pickup_datetime'].dt.year
train['pickup_datetime_day_of_week_name'] = train['pickup_datetime'].dt.weekday_name
train['pickup_datetime_day_of_week'] = train['pickup_datetime'].dt.weekday
train['pickup_datetime_day_of_hour'] = train['pickup_datetime'].dt.hour
train.head(7)
business_train = train[train['pickup_datetime_day_of_week'] < 5 ]
business_train.head(5)
early_business_hours = business_train[business_train['pickup_datetime_day_of_hour'] < 10]
late_business_hours = business_train[business_train['pickup_datetime_day_of_hour'] > 6]
data = [go.Scattermapbox(
            lat= early_business_hours['dropoff_latitude'] ,
            lon= early_business_hours['dropoff_longitude'],
            customdata = early_business_hours['key'],
            mode='markers',
            marker=dict(
                size= 5,
                color = 'gold',
                opacity = .8),
            name ='early_business_hours'
          ),
        go.Scattermapbox(
            lat= late_business_hours['dropoff_latitude'] ,
            lon= late_business_hours['dropoff_longitude'],
            customdata = late_business_hours['key'],
            mode='markers',
            marker=dict(
                size= 5,
                color = 'cyan',
                opacity = .8),
            name ='late_business_hours'
          )]
layout = go.Layout(autosize=False,
                   mapbox= dict(accesstoken="pk.eyJ1Ijoic2hhejEzIiwiYSI6ImNqYXA3NjhmeDR4d3Iyd2w5M2phM3E2djQifQ.yyxsAzT94VGYYEEOhxy87w",
                                bearing=10,
                                pitch=60,
                                zoom=13,
                                center= dict(
                                         lat=40.721319,
                                         lon=-73.987130),
                                style= "mapbox://styles/shaz13/cjiog1iqa1vkd2soeu5eocy4i"),
                    width=900,
                    height=600, title = "Early vs. Late Business Days Pickup Locations")
fig = dict(data=data, layout=layout)
iplot(fig)
weekend_train  = train[train['pickup_datetime_day_of_week'] >= 5 ]
early_weekend_hours = weekend_train[weekend_train['pickup_datetime_day_of_hour'] < 10]
late_weekend_hours = weekend_train[weekend_train['pickup_datetime_day_of_hour'] > 6]
data = [go.Scattermapbox(
            lat= early_weekend_hours['dropoff_latitude'] ,
            lon= early_weekend_hours['dropoff_longitude'],
            customdata = early_weekend_hours['key'],
            mode='markers',
            marker=dict(
                size= 5,
                color = 'violet',
                opacity = .8),
            name ='early_weekend_hours'
          ),
        go.Scattermapbox(
            lat= late_weekend_hours['dropoff_latitude'] ,
            lon= late_weekend_hours['dropoff_longitude'],
            customdata = late_weekend_hours['key'],
            mode='markers',
            marker=dict(
                size= 5,
                color = 'orange',
                opacity = .8),
            name ='late_weekend_hours'
          )]
layout = go.Layout(autosize=False,
                   mapbox= dict(accesstoken="pk.eyJ1Ijoic2hhejEzIiwiYSI6ImNqYXA3NjhmeDR4d3Iyd2w5M2phM3E2djQifQ.yyxsAzT94VGYYEEOhxy87w",
                                bearing=10,
                                pitch=60,
                                zoom=13,
                                center= dict(
                                         lat=40.721319,
                                         lon=-73.987130),
                                style= "mapbox://styles/shaz13/cjiog1iqa1vkd2soeu5eocy4i"),
                    width=900,
                    height=600, title = "Early vs. Late Weekend Days Pickup Locations")
fig = dict(data=data, layout=layout)
iplot(fig)
high_fares = train[train['fare_amount'] > train.fare_amount.mean() + 3* train.fare_amount.std()]
high_fares.head()

data = [go.Scattermapbox(
            lat= high_fares['pickup_latitude'] ,
            lon= high_fares['pickup_longitude'],
            customdata = high_fares['key'],
            mode='markers',
            marker=dict(
                size= 8,
                color = 'violet',
                opacity = .8),
            name ='high_fares_pick_up'
          ),
        go.Scattermapbox(
            lat= high_fares['dropoff_latitude'] ,
            lon= high_fares['dropoff_longitude'],
            customdata = high_fares['key'],
            mode='markers',
            marker=dict(
                size= 8,
                color = 'gold',
                opacity = .8),
            name ='high_fares_drop_off'
          )]
layout = go.Layout(autosize=False,
                   mapbox= dict(accesstoken="pk.eyJ1Ijoic2hhejEzIiwiYSI6ImNqYXA3NjhmeDR4d3Iyd2w5M2phM3E2djQifQ.yyxsAzT94VGYYEEOhxy87w",
                                bearing=10,
                                pitch=60,
                                zoom=13,
                                center= dict(
                                         lat=40.721319,
                                         lon=-73.987130),
                                style= "mapbox://styles/shaz13/cjk4wlc1s02bm2smsqd7qtjhs"),
                    width=900,
                    height=600, title = "High Fare Locations")
fig = dict(data=data, layout=layout)
iplot(fig)