# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px #plotly graph
import plotly.io as pio #plotly graph

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Reading data from dataset
covid_data = pd.read_csv('/kaggle/input/covid19-bangladesh-dataset/COVID-19-Bangladesh.csv', parse_dates=['Date'])
#Print raw dataset
covid_data.head()
# Active Case = confirmed - deaths - recovered
covid_data['active'] = covid_data['Confirmed'].cumsum() - covid_data['Deaths'].cumsum() - covid_data['Recovered'].cumsum()

# Quarantine cases 
q_cases = ['Quarantine', 'Released From Quarantine']
covid_data[q_cases] = covid_data[q_cases].fillna(0).astype(int)

#print dataset with active cases
covid_data.tail()
#theme customization
pio.templates.default = "plotly_dark"

#Groping of Confirmed, Recovered, Deaths and active cases by Date
group = covid_data.groupby('Date')['Date', 'Confirmed', 'Recovered', 'Deaths', 'active'].sum().reset_index()

#Plotting Confirmed cases
fig = px.line(group, x="Date", y="Confirmed", 
              title="National Confirmed Cases Over Time")

fig.show()

#Plotting Recovered cases
fig = px.line(group, x="Date", y="Recovered", 
              title="National Recovered Cases Over Time")

fig.show()

#Plotting Death cases
fig = px.line(group, x="Date", y="Deaths", 
              title="National Deaths Over Time")

fig.show()

#Plotting Active cases
fig = px.line(group, x="Date", y="active", 
              title="National Active Cases Over Time")

fig.show()
#Bangladesh Population Number(Source: Worldometer)
bd_population = 164286984

#Calculating infectionRate, mortalityRate and recoveryRate accross Population
covid_data['infectionRate'] = round((covid_data['Confirmed'].cumsum()/bd_population)*100, 5)
covid_data['mortalityRate'] = round((covid_data['Deaths'].cumsum()/covid_data['Confirmed'].cumsum())*100, 2)
covid_data['recoveryRate'] = round((covid_data['Recovered'].cumsum()/covid_data['Confirmed'].cumsum())*100, 2)

#Printing formatted dataset
covid_data.tail()
#Groping of infectionRate, mortalityRate, recoveryRate by Date
group = covid_data.groupby('Date')['infectionRate', 'mortalityRate', 'recoveryRate'].sum().reset_index()

#Plotting infectionRate cases
fig = px.line(group, x="Date", y="infectionRate", 
              title="National Infection Rate Over Time")

fig.show()

#Plotting mortalityRate cases
fig = px.line(group, x="Date", y="mortalityRate", 
              title="National Mortality Rate Over Time")

fig.show()

#Plotting recoveryRate cases
fig = px.line(group, x="Date", y="recoveryRate", 
              title="National Recovery Rate Over Time")

fig.show()
#formatted dataset details
covid_data.info()