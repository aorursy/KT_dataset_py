# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import plotly.express as px
import plotly.graph_objects as go
import folium
import os
import json
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows',20000, 'display.max_columns',100)
corona_data = pd.read_csv("/kaggle/input/covid19-in-italy/covid19_italy_region.csv")
corona_data.head()
#detail information about each column whether it conatins null value or non null
corona_data.info()
#Number of cases in a particular region
data_by_region = corona_data.groupby("RegionName")[['TotalPositiveCases', 'Deaths', 'Recovered','TestsPerformed', 'HospitalizedPatients']].sum().reset_index()
data_by_region.head()
#Representation of number of cases in a region on bar chart
px.bar(data_frame= (data_by_region).sort_values('TotalPositiveCases')
       , x='RegionName'
       , y='TotalPositiveCases'
       , template='ggplot2'
       , color='RegionName'
       , log_y= 'True'
       , title='Number of Positive Cases vs Region Name'
      )
#Representation of number of death in a region on bar chart
px.bar(data_frame= (data_by_region).sort_values('Deaths')
       , x='RegionName'
       , y='Deaths'
       , template='ggplot2'
       , color='RegionName'
       , log_y= 'True'
       , title='Number of Death vs Region Name'
      )
#Representation of recovery in a region on bar chart
px.bar(data_frame= (data_by_region).sort_values('Recovered')
       , x='RegionName'
       , y='Recovered'
       , template='ggplot2'
       , color='RegionName'
       , log_y= 'True'
       , title='Recovery vs Region Name'
      )
#test performed vs region
px.bar(data_frame= (data_by_region).sort_values('TestsPerformed')
       , x='RegionName'
       , y='TestsPerformed'
       , template='ggplot2'
       , color='RegionName'
       , log_y= 'True'
       , title='Tests Performed vs Region Name'
      )
       
#HospitalizedPatients  vs region
px.bar(data_frame= (data_by_region).sort_values('HospitalizedPatients')
       , x='RegionName'
       , y='HospitalizedPatients'
       , template='ggplot2'
       , color='RegionName'
       , log_y= 'True'
       , title='HospitalizedPatients vs Region Name'
      )
#Test & Confirm Cases vs Region
plt.figure(figsize=(29,15))
plt.bar(corona_data.RegionName, corona_data.TestsPerformed,label="Tests Performed")
plt.bar(corona_data.RegionName, corona_data.TotalPositiveCases,label="Confirm Cases")
plt.xlabel('Region')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=18, )
plt.title('Test & Confirm Cases vs Region',fontsize = 35)

plt.show()
#Confirm Cases & People Hospitalised vs Region
plt.figure(figsize=(25,12))
plt.bar(corona_data.RegionName, corona_data.TotalPositiveCases,label="Confirm Cases")
plt.bar(corona_data.RegionName, corona_data.TotalHospitalizedPatients,label="Hospitalized Patients")

plt.xlabel('Region')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Confirm Cases & People Hospitalised vs Region',fontsize= 35)
plt.show()
#Death & Recovery vs Region
plt.figure(figsize=(25,12))
plt.bar(corona_data.RegionName, corona_data.Recovered,label="Recovery")
plt.bar(corona_data.RegionName, corona_data.Deaths,label="Death")
plt.xlabel('Region')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=16)
plt.title('Death & Recovery vs Region', fontsize= 35)
plt.show()
#Daily Tests Performed in Italy Datewise
covid_data = corona_data.groupby(['Date'])['TotalPositiveCases', 'TestsPerformed', 'Deaths'].sum().reset_index().sort_values('Date', ascending=True)
covid_data['Daily TestsPerformed'] = covid_data['TestsPerformed'].sub(covid_data['TestsPerformed'].shift())
covid_data['Daily TestsPerformed'].iloc[0] = covid_data['TestsPerformed'].iloc[0]
covid_data['Daily TestsPerformed'] = covid_data['Daily TestsPerformed'].astype(int)
fig = px.bar(covid_data, y='Daily TestsPerformed', x='Date',hover_data =['Daily TestsPerformed'], color='Daily TestsPerformed', height=500)
fig.update_layout(
    title='Daily Tests Performed in Italy Datewise')
fig.show()
#Daily Total Positive Cases in Italy Datewise
covid_data['Daily Cases'] = covid_data['TotalPositiveCases'].sub(covid_data['TotalPositiveCases'].shift())
covid_data['Daily Cases'].iloc[0] = covid_data['TotalPositiveCases'].iloc[0]
covid_data['Daily Cases'] = covid_data['Daily Cases'].astype(int)
fig = px.bar(covid_data, y='Daily Cases', x='Date',hover_data =['Daily Cases'], color='Daily Cases', height=500)
fig.update_layout(
    title='Daily Positive Cases in Italy Datewise')
fig.show()
#Daily Deaths in Italy Datewise
covid_data['Daily Deaths'] = covid_data['Deaths'].sub(covid_data['Deaths'].shift())
covid_data['Daily Deaths'].iloc[0] = covid_data['Deaths'].iloc[0]
covid_data['Daily Deaths'] = covid_data['Daily Deaths'].astype(int)
fig = px.bar(covid_data, y='Daily Deaths', x='Date',hover_data =['Daily Deaths'], color='Daily Deaths', height=500)
fig.update_layout(
    title='Daily Deaths in Italy Datewise')
fig.show()
covid_data = corona_data.groupby(['Date'])['TotalPositiveCases', 'CurrentPositiveCases', 'Deaths'].sum().reset_index().sort_values('Date', ascending=False)
covid_data['Mortality Rate'] = ((covid_data['Deaths']/covid_data['TotalPositiveCases'])*100)
fig = go.Figure()
fig.add_trace(go.Scatter(x=covid_data['Date'], y=covid_data['Mortality Rate'], mode='lines + markers', name ='TotalPositiveCases', marker_color = 'red'))
fig.update_layout(title_text = 'Corona Mortality Rate in Italy', plot_bgcolor='rgb(225,230,255)')
fig.show()