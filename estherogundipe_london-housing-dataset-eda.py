# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd 

import geopandas as gpd 

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

from plotly.offline import init_notebook_mode, iplot, plot

import plotly.graph_objs as go



import folium

from folium import Choropleth, Circle, Marker

from folium import plugins
# Load the Data

lon_month = pd.read_csv('/kaggle/input/housing-in-london/housing_in_london_monthly_variables.csv')
# To determine the number of observations and total entries

lon_month.shape
display('London Housing Data at year level')

display(lon_month.head(5))
# Information about number of rows/observations, columns/features and missing values if any.

lon_month.info()
# Computes a summary of statistics pertaining to the DataFrame columns.

lon_month.describe()
# MISSING VALUES

def compute_missing_values(df):

    #total number of missing values

    total_missing = df.isnull().sum().sort_values(ascending=False)

    

    #calculating the percentage of missing values

    percentage_missing = (100 * df.isnull().sum() / len(df))

    

    #Missing values table - total, percentage

    table_missing = pd.concat([total_missing, percentage_missing], axis = 1, 

                              keys = ['Missing values', 'Percentage of Missing Values'])

    

    #Filtering the columns with missing values

    table_missing = table_missing[table_missing.iloc[:, 0] != 0]

    

    #Summary 

    print("Total number of columns:" + str(df.shape[1]) + "\nColumns with missing values:" +str(table_missing.shape[1]))

    

    return table_missing



missing_values = compute_missing_values(lon_month)

missing_values.style.background_gradient(cmap='Reds')
# Replacing the missing values with the mean of the column

no_of_crimes_mean= lon_month['no_of_crimes'].mean()

lon_month = lon_month.fillna({'no_of_crimes' : no_of_crimes_mean})



# Removing the missing values

lon_month = lon_month[lon_month['houses_sold'].notna()]

lon_month.isnull().sum()
# Filtering intial data by checking if the area is a London Borough to obtain a subset.

london = lon_month[lon_month['borough_flag'] == 1]

london.head()
london['approx_house_price'] = london['average_price'].mul(london['houses_sold'])
# Find the unique elements of an array

london.area.unique()
# Compute the mean

london_mean = london.groupby('area').mean().reset_index()

london_mean.head()
# Convert string Date time into Python Date time object. 

lon_month.date = pd.to_datetime(lon_month.date)
lon_month.tail()
# Line Chart that shows the Average Price by Area Trend

px.line(lon_month,x='date',y='average_price',color='area',title='Average Price by Area Trend')
# Specific Area Trend

kc_2019 = lon_month[(lon_month.area == 'kensington and chelsea') & (lon_month.date.dt.year == 2019)].average_price.mean()

kc_1995 = lon_month[(lon_month.area == 'kensington and chelsea') & (lon_month.date.dt.year == 1995)].average_price.mean()

display(round(kc_2019 - kc_1995,2), str(round((kc_2019 / kc_1995 * 100),2)) + '%')
# Line Chart of Average Housing Price by Borough_Flag Trend

px.line(lon_month.groupby(['date','borough_flag']).average_price.mean().to_frame().reset_index(),x='date',y='average_price',color='borough_flag',title='Average Housing Price by Borough_Flag Trend')
# Line Chart of Number of Houses Sold

fig = px.line(london,x='date',y='houses_sold',color='area',title='Number of Houses Sold')

fig.update_layout(xaxis_type="date", yaxis_type="log", xaxis_title='Year', yaxis_title='Price (£)')

fig.show()
lon_month
# Bar Chart of the Area of Borough Flag

px.bar(lon_month.groupby("borough_flag").area.nunique().reset_index(),x="borough_flag",y="area")
# Bar Chart of Houses Sold by the Area

houses_2019 = lon_month[lon_month.date.dt.year == 2019].groupby("area").houses_sold.sum().reset_index()

px.bar(houses_2019,x="area",y="houses_sold")
# Total Value of Houses Sold

lon_month['total'] = lon_month.average_price * lon_month.houses_sold

total_2019 = lon_month[lon_month.date.dt.year == 2019].groupby('area').total.sum().reset_index()

print("Total value was %s GBP" % '{:,.2f}'.format(total_2019.total.sum()))
# Bar Chart of Area Price and Number of houses sold * 1000 per Borough

trace1 = go.Bar(

                x = london_mean.area,

                y = london_mean.houses_sold*1000,

                name = "Number of houses sold * 1000",

                marker = dict(color = 'rgba(8, 103, 103, 0.8)',

                             line=dict(color='rgb(0,0,0)',width=1)),

                text = london_mean.area)

trace2 = go.Bar(

                x = london_mean.area,

                y = london_mean.average_price,

                name = "Average Price (£)",

                marker = dict(color = 'rgba(103, 8, 8, 0.7)',

                             line=dict(color='rgb(0,0,0)',width=1)),

                text = london_mean.area)



data = [trace1,trace2]

layout = go.Layout(barmode = "group", title="Average Price and Number of houses sold * 1000 per Borough")

fig = go.Figure(data = data, layout = layout)

fig.update_xaxes(ticks="outside", tickwidth=2,tickangle=45, ticklen=10,title_text="Boroughs of London")

iplot(fig)
px.pie(total_2019,values='total',names='area')