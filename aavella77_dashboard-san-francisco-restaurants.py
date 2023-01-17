import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import sys
#!{sys.executable} -m pip install csvvalidator

# Number of inspections to show.  By default change the latest 100 inspections
MAX_RECORDS = 100

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os

data = pd.read_csv(r'../input/restaurant-scores-lives-standard.csv', parse_dates=['inspection_date'])

# Discard rows with blank columns (NaN values)
data = data.dropna()

# Sort data by inspection date
data = data.sort_values(by=['inspection_date'])

#Troubleshooting lines
#print(os.listdir("../input"))
#print("Fields: " )
#print(data.columns.values)

# Use the last 100 inspections
#data.tail(MAX_RECORDS)


import folium

SF_COORDINATES = (37.76, -122.45)

# create empty map zoomed in on San Francisco
map = folium.Map(location=SF_COORDINATES, zoom_start=12) 

# add a marker for every record in the filtered data, use a clustered view
for each in data[0:MAX_RECORDS].iterrows():
    folium.Marker([each[1]["business_latitude"],
                   each[1]["business_longitude"]],
                  popup=each[1]["business_name"] + " Violation: " 
                  + each[1]["violation_description"]).add_to(map)
    
display(map)
#print(sf_restaurants_data['inspection_date'])
data = data.tail(MAX_RECORDS)

data = data.rename(columns={'inspection_date': 'Date'})
data.columns

data['Date'] = pd.to_datetime(data['Date'], format = "%Y-%m-%d")

# count of meets per month
meets_by_month = data['Date'].groupby([data.Date.dt.year, data.Date.dt.month, data.Date.dt.day]).agg('count') 

# convert to dataframe
meets_by_month = meets_by_month.to_frame()

# move date month from index to column
meets_by_month['date'] = meets_by_month.index

# rename column
meets_by_month = meets_by_month.rename(columns={meets_by_month.columns[0]:"meets"})

# re-parse dates
meets_by_month['date'] = pd.to_datetime(meets_by_month['date'], format="(%Y, %m, %d)")

# remove index
meets_by_month = meets_by_month.reset_index(drop=True)

# get month of meet
meets_by_month['month'] = meets_by_month.date.dt.month




# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data1 = [go.Bar(x=meets_by_month.date, y=meets_by_month.meets)]

# specify the layout of our figure
layout = dict(title = "Number Inspections per Day",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data1, layout = layout)
iplot(fig)
data = pd.read_csv(r'../input/restaurant-scores-lives-standard.csv')

# Discard rows with blank columns (NaN values)
data = data.dropna()
#print(type(data))

data = data.tail(MAX_RECORDS)
#print(data.columns)

#print(data['risk_category'])

values_counts = data['risk_category'].value_counts()
labels_risk_categories = list(data['risk_category'].value_counts().index)
              
#print(values_counts) 
#print(labels_risk_categories)              


# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data1 = [go.Pie(labels=labels_risk_categories, values=values_counts)]

# specify the layout of our figure
layout = dict(title = "Inspections Risk Categories")

# create and show our figure
fig = dict(data = data1, layout = layout)
iplot(fig)
