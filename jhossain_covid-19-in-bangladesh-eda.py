import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # static data visulization 

import matplotlib.dates as mdates # datetime format 

import seaborn as sns # data visualization 

from datetime import datetime, timedelta

import plotly.express as px 



# Color pallate 

cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801' # active case - yellow


import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Read and store dataset into dataframe called df 

df = pd.read_csv("/kaggle/input/covid19-in-bangladesh/COVID-19_in_bd.csv")
# Examine first few rows 

df.head()
# Check shape of dataset 

df.shape 
# listing columns or variables 

df.columns
# Numerical summary 

df.describe()
# Basic info about dataset 

df.info()
# Check missing values 

df.isnull().sum()
# Find starting date of outbreak in Bangladesh 

start = df['Date'].min()

print(f"Starting date of COVID-19 outbreak in Bangladesh is {start}")
# Cumulative confirm cases 

confirmed = df['Confirmed'].max()

# Cumulative deaths cases 

deaths = df['Deaths'].max()

# Cumulative recovered cases 

recovered = df['Recovered'].max()

# Active cases = confirmed - deaths - recovered 

active = df['Confirmed'].max() - df['Deaths'].max() - df['Recovered'].max()

# Printing the result 

print(f"Cumulative Confirmed Cases = {confirmed}")

print(f"Cumulative Deaths Cases = {deaths}")

print(f"Cumulative Recovered Cases = {recovered}")

print(f"Active Cases = {active}")
# Create a new column for active case 

df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
# Now take a look at dataset 

df.head()
# Grouping cases by date 

temp = df.groupby('Date')['Confirmed', 'Recovered', 'Deaths', 'Active'].sum().reset_index() 

# Unpivoting 

temp = temp.melt(id_vars='Date',value_vars = ['Confirmed', 'Recovered', 'Deaths', 'Active'], var_name='Case', value_name='Count') 



# Visualization

fig = px.line(temp, x='Date', y='Count', color='Case', color_discrete_sequence=[cnf, rec, dth, act], template='ggplot2') 

fig.update_layout(title="COVID-19 Cases Over Time in Bangladesh")

fig.show()
# Grouping cases by date 

temp = df.groupby('Date')['Confirmed', 'Recovered', 'Deaths', 'Active'].sum().reset_index() 

# Unpivoting 

temp = temp.melt(id_vars='Date',value_vars = ['Confirmed', 'Recovered', 'Deaths', 'Active'], var_name='Case', value_name='Count') 



# Visualization

fig = px.area(temp, x='Date', y='Count', color='Case', color_discrete_sequence=[cnf, rec, dth, act], template='ggplot2') 

fig.update_layout(title="COVID-19 Cases Over Time in Bangladesh", xaxis_rangeslider_visible=True)

fig.show()