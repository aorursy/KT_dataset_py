# datetime operations

from datetime import timedelta



# for numerical analyiss

import numpy as np



# to store and process data in dataframe

import pandas as pd



# basic visualization package

import matplotlib.pyplot as plt



# advanced ploting

import seaborn as sns



# interactive visualization

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots



# for offline ploting

from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)



# hide warnings

import warnings

warnings.filterwarnings('ignore')



# color pallette

# Hexademical code RRGGBB (True Black #000000, True White #ffffff)

cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 



# list files

!ls ../input/corona-virus-report
# Country wise

country_wise = pd.read_csv('../input/corona-virus-report/country_wise_latest.csv')



# Replace missing values '' with NAN and then 0

country_wise = country_wise.replace('', np.nan).fillna(0)



# Grouped by day, country

full_grouped = pd.read_csv('../input/corona-virus-report/full_grouped.csv')



# Convert Date from Dtype "Object" (or String) to Dtype "Datetime"

full_grouped['Date'] = pd.to_datetime(full_grouped['Date'])



# Day wise

day_wise = pd.read_csv('../input/corona-virus-report/day_wise.csv')

day_wise['Date'] = pd.to_datetime(day_wise['Date'])
# Use Boolean indexing to generate a mask which is just a series of boolean values representing whether the column contains the specific element or not

selected = full_grouped['Country/Region'].str.contains('Japan')



# Apply this mask to our original DataFrame to filter the required values.

japan = full_grouped[selected]

japan["New active"] = japan["Active"].diff()



#Melting Adjustments

temp = japan.melt(id_vars="Date", value_vars=['New cases', 'New deaths'],

                 var_name='Case', value_name='Count')



japan.info()

japan.head(10)

japan.describe()



#Figure Creation

fig = px.area(temp, x="Date", y="Count", color='Case', height=600, width=1200,

             title='Cases over time in Japan', color_discrete_sequence = [rec, dth, act])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
# Use Boolean indexing to generate a mask which is just a series of boolean values representing whether the column contains the specific element or not

selected = full_grouped['Country/Region'].str.contains('China')



# Apply this mask to our original DataFrame to filter the required values.

china = full_grouped[selected]

china["New active"] = china["Active"].diff()



#Melting Adjustments

temp1 = china.melt(id_vars="Date", value_vars=['New cases'],

                 var_name='Case', value_name='Count')



#Figure Creation

fig = px.area(temp1, x="Date", y="Count", color='Case', height=600, width=1200,

             title='Cases over time in China', color_discrete_sequence = [rec, dth, act])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
# Use Boolean indexing to generate a mask which is just a series of boolean values representing whether the column contains the specific element or not

selected = full_grouped['Country/Region'].str.contains('US')



# Apply this mask to our original DataFrame to filter the required values.

us = full_grouped[selected]

us["New active"] = us["Active"].diff()



#Melting Adjustments

temp2 = us.melt(id_vars="Date", value_vars=['New cases'],

                 var_name='Case', value_name='Count')



#Figure Creation

fig = px.area(temp2, x="Date", y="Count", color='Case', height=600, width=1200,

             title='Cases over time in USA', color_discrete_sequence = [rec, dth, act])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
# Use Boolean indexing to generate a mask which is just a series of boolean values representing whether the column contains the specific element or not

selected = full_grouped['Country/Region'].str.contains('Taiwan')



# Apply this mask to our original DataFrame to filter the required values.

taiwan = full_grouped[selected]

taiwan["New active"] = taiwan["Active"].diff()



#Melting Adjustments

temp3 = taiwan.melt(id_vars="Date", value_vars=['New cases'],

                 var_name='Case', value_name='Count')



#Figure Creation

fig = px.area(temp3, x="Date", y="Count", color='Case', height=600, width=1200,

             title='Cases over time in Taiwan', color_discrete_sequence = [rec, dth, act])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
# Use Boolean indexing to generate a mask which is just a series of boolean values representing whether the column contains the specific element or not

selected = full_grouped['Country/Region'].str.contains('Korea')



# Apply this mask to our original DataFrame to filter the required values.

korea = full_grouped[selected]

korea["New active"] = korea["Active"].diff()



#Melting Adjustments

temp4 = korea.melt(id_vars="Date", value_vars=['New cases'],

                 var_name='Case', value_name='Count')



#Figure Creation

fig = px.area(temp4, x="Date", y="Count", color='Case', height=600, width=1200,

             title='Cases over time in Korea', color_discrete_sequence = [rec, dth, act])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()