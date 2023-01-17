# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

# Any results you write to the current directory are saved as output.

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

tripduration, starttime, stoptime, start_station_id, start_station_name, start_station_latitude,

start_station_longitude, end_station_id, end_station_name, end_station_latitude,

end_station_longitude, bikeid, usertype, birth_year, gender, extract (MONTH from starttime) as month



FROM

  `bigquery-public-data.new_york.citibike_trips`

  WHERE CAST(starttime as STRING) LIKE "2016%"  

order by month asc

LIMIT 10000

;

"""



data = nyc.query_to_pandas_safe(query, max_gb_scanned=10)



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

pd.set_option('max_colwidth',50)

data.to_csv('1-10k_citibike_2016.csv',index=False)