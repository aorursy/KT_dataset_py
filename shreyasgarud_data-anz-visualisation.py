# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
import plotly as pty
import plotly.offline as pyo
import plotly.graph_objs as go
# Set notebook mode to work in offline
pyo.init_notebook_mode()


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
transaction = pd.read_csv("/kaggle/input/anz-synthesised-transaction-dataset/anz.csv")
# Overview of data
transaction.head()
# to check columns in data
transaction.info()
transaction['status'].unique()
# The visualisation represents the amount of transaction done by females and mals based on status. 

import plotly.express as px
px.bar(transaction,x='status',y='amount',color='gender',barmode='group')
# Amount of transaction date wise
px.line(transaction.groupby(['date']).sum()['amount'])
# average amount of transaction
px.line(transaction.groupby(['date']).mean()['amount'])
# mean amount of transactions executed per day
transaction.groupby(['date']).sum()['amount'].mean()
# Statewise representation of amount 

px.bar(transaction,x=transaction['merchant_state'].dropna().unique(),y=transaction.groupby(['merchant_state']).sum()['amount'])
# Convert the date column from string to datetime type
transaction['date']=pd.to_datetime(transaction['date'])
# To get the mean amount of transactions per month
transaction.groupby(transaction['date'].dt.strftime('%B'))['amount'].mean()
transaction.groupby(transaction['date'].dt.strftime('%B'))['amount'].count()
transaction['long_lat'].head()
# Representation of transaction data on map

lat = []
lon = []

# For each row in a varible,
for row in transaction['long_lat']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to long
        lon.append(row.split(' ')[0])
        # Split the row by comma and append
        # everything after the comma to lat
        lat.append(row.split(' ')[1])
    # But if you get an error
    except:
        # append a missing value to lat
        lon.append(np.NaN)
        # append a missing value to lon
        lat.append(np.NaN)

# Create two new columns from lat and lon
transaction['latitude'] = lat
transaction['longitude'] = lon
transaction['latitude'].head()
transaction['latitude']=pd.to_numeric(transaction['latitude'])
transaction['longitude']=pd.to_numeric(transaction['longitude'])
import math
map = folium.Map(location=['-25.2744','133.7751'],tiles='cartodbpositron',zoom_start=5)


mc = MarkerCluster()
for idx, row in transaction.iterrows():
    if not math.isnan(row['longitude']) and not math.isnan(row['latitude']):
        mc.add_child(Marker([row['latitude'], row['longitude']]))
map.add_child(mc)



map
