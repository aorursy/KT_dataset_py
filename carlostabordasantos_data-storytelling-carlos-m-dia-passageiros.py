import matplotlib.pyplot as plt

import seaborn as sns

import folium

import os

import bq_helper

import csv

import requests



from bq_helper import BigQueryHelper



nyc = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="new_york")



query = """SELECT 

pickup_datetime, 

dropoff_datetime,

passenger_count,

trip_distance * 1.6 as trip_distance,

pickup_longitude,

pickup_latitude,

dropoff_longitude,

dropoff_latitude,

payment_type,

total_amount

FROM

  `bigquery-public-data.new_york.tlc_yellow_trips_2016`

  

where pickup_longitude is not null

and dropoff_longitude is not null

and trip_distance >0



LIMIT 100000



;

"""

data = nyc.query_to_pandas_safe(query, max_gb_scanned=20)

import pandas as pd

data.head()
data['total_amount'].sum() / data['passenger_count'].count()
data['total_amount'].sum() / data['trip_distance'].sum()
import numpy as np
data['trip_distancebin']=pd.cut(data['trip_distance'], bins=[0,5,10,15,20,30,np.inf],right=False)

data.head()
kmedio=data.groupby(by='trip_distancebin')['total_amount'].mean()

kmedio
kmedio.plot.line()
data['teste']=(data['total_amount']> 1) & (data['total_amount']<50)

data.head()
kmedio = kmedio['trip_distancebin']

kmedio.head(10)
kmedio_pivot = pd.pivot_table(data=kmedio_pivot,

                       index='trip_distancebin',

                       values='total_amount',

                       columns='payment_type')

kmedio_pivot