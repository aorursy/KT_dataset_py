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

from IPython.core.display import HTML
import folium
import datetime
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import lxml.html as lh
import pandas as pd
import re
import time
import psutil
import json

import numpy as np
from PIL import Image
import os
from os import path
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from pandas.plotting import register_matplotlib_converters
import plotly.express as px
from IPython.display import display, Markdown, Latex
import matplotlib as plot
from matplotlib.pyplot import figure
import seaborn as sns

register_matplotlib_converters()
from IPython.display import Markdown


dataset = pd.DataFrame()
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
df.head()

data_df = df.copy()
data_df['Date'] = data_df['Date'].apply(pd.to_datetime)
data_df.drop(['Sno', 'Time'],axis=1,inplace=True)
data_df.head()


from datetime import date
data_april = data_df[data_df['Date'] > pd.Timestamp(date(2020,4,12))]

state_cases = data_april.groupby('State/UnionTerritory')['Confirmed','Deaths','Cured'].max().reset_index()
state_cases['Active'] = state_cases['Confirmed'] - (state_cases['Deaths']- state_cases['Cured'])
state_cases["Death Rate (per 100)"] = np.round(100*state_cases["Deaths"]/state_cases["Confirmed"],2)
state_cases["Cure Rate (per 100)"] = np.round(100*state_cases["Cured"]/state_cases["Confirmed"],2)
state_cases
import requests
import plotly.graph_objects as go
import re
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
LiveJson = 'https://api.covid19india.org/data.json'
r = requests.get(LiveJson)
Data_india = r.json()

total = []
recovered = []
deseased = []
timeStamp = []
for index in range(len(Data_india['cases_time_series'])):
    total.append(int(re.sub(',','',Data_india['cases_time_series'][index]['totalconfirmed'])))
    recovered.append(int(re.sub(',','',Data_india['cases_time_series'][index]['totalrecovered'])))
    deseased.append(int(re.sub(',','',Data_india['cases_time_series'][index]['totaldeceased'])))
    
    timeStamp.append(Data_india['cases_time_series'][index]['date'])
fig = go.Figure()
fig = fig.add_trace(go.Scatter(x = timeStamp , y = total,mode = 'lines+markers',name = 'Confirmed Cases'))
fig = fig.add_trace(go.Scatter(x = timeStamp , y = recovered,mode = 'lines+markers',name = 'Recovered Cases'))
fig = fig.add_trace(go.Scatter(x = timeStamp , y = deseased,mode = 'lines+markers',name = 'Deseaced Cases'))

fig = fig.update_layout(title = "India Covid19",xaxis_title = "Date",yaxis_title="Number")
fig.show()
# fig = go.Figure()
# fig = fig.add_trace(go.Bar(x = timeStamp ,y = total,mode = "lines+markers" ))
# fig.show()

fig = go.Figure([go.Bar(x=timeStamp, y=total)])
fig.show()
fig = go.Figure([go.Bar(x=timeStamp, y=recovered)])
fig.show()
Hospitalbeds = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')
Hospitalbeds = Hospitalbeds[:-1]


states = []
active = []

for index in range(len(Data_india['statewise'])):
    if index == 0:
        continue
    states.append(str(re.sub(',','',Data_india['statewise'][index]['state'])))
    active.append(int(re.sub(',','',Data_india['statewise'][index]['active'])))

indiaActive = pd.DataFrame()  
indiaActive['States'] = states
indiaActive['active'] = active



state_cases['State/UnionTerritory'] = state_cases['State/UnionTerritory'].apply(lambda x: re.sub(' and ',' & ',x))
activelist = []
for state in Hospitalbeds['State/UT'].tolist():
    try:
        activelist.append(indiaActive[indiaActive['States'] == state]['active'].values[0])
    except:
        try:
            activelist.append(state_cases[state_cases['State/UnionTerritory'] == state]['Active'].values[0])
        except:
            activelist.append(0)

    
Hospitalbeds['active'] = activelist

fig = go.Figure(data=[go.Bar(
            y= (Hospitalbeds['NumRuralBeds_NHP18']+Hospitalbeds['NumUrbanBeds_NHP18']).tolist(), 
            x=Hospitalbeds['State/UT'].tolist(),
            name='Beds availible in states',
            marker_color='#000000'),
            
            go.Bar(
            y=Hospitalbeds['active'].tolist(), 
            x=Hospitalbeds['State/UT'].tolist(),
            name='Positve Cases',
            marker_color='#FF0000')
                     ])

# Change the bar mode
fig.update_layout(barmode='stack', template="ggplot2", title_text = '<b>Sample Tested for COVID-19 in India (Day Wise)</b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))
fig.show()