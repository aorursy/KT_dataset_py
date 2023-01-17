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

from datetime import datetime
from collections import defaultdict
import requests
import operator
import pandas as pd
import matplotlib.pyplot as plt

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
r = requests.get('https://api.covid19india.org/raw_data.json')
jsonData = r.json()
listJsonData = jsonData.get('raw_data')
Coviddf = pd.DataFrame.from_dict(listJsonData, orient='columns')
RawCoviddf= Coviddf
RawCoviddf['dateannounced'] = pd.to_datetime(RawCoviddf['dateannounced'],format ='%d/%m/%Y')
RawCoviddf.dropna(subset=['detectedstate'],inplace = True)
url1 = 'https://api.covid19india.org/states_daily_csv/recovered.csv'
recDf = pd.read_csv(url1)
recDf = recDf.iloc[:,0:recDf.shape[1]-1]
recDf.fillna(0,inplace = True)
url2 = 'https://api.covid19india.org/states_daily_csv/deceased.csv'
decDf =pd.read_csv(url2)
decDf = decDf.iloc[:,0:decDf.shape[1]-1]
decDf.fillna(0,inplace = True)
url3 = 'http://api.covid19india.org/states_daily_csv/confirmed.csv'
conDf = pd.read_csv(url3)
conDf = conDf.iloc[:,0:conDf.shape[1]-1]
conDf.fillna(0,inplace = True)
#calculation Part
value = list(Tabulation.sum(axis=0,skipna=True))

# display using plotly pie chart
colors = ['red','green','grey']
fig = go.Figure(data=go.Pie(values = value[:-1], labels = Tabulation.columns.tolist(),textinfo='percent+label',marker=dict(colors=colors)))
fig.update_layout(title_text='Affect of COVID 19 in India')

fig.show()
df = combinedDf.groupby('Date').sum().groupby('Date').cumsum()
df['Confirmed'] = df['Confirmed'] - (df['Recovered']+df['Dead'])
df.columns =['Active','Recovered','Dead']

#display
fig = go.Figure(data=[
    go.Bar(name='Dead', x=df.index.tolist(), y=df['Dead'],marker_color = 'grey'),
    go.Bar(name='Recovered', x=df.index.tolist(), y=df['Recovered'],marker_color = 'green'),
    go.Bar(name='Active', x=df.index.tolist(), y=df['Active'],marker_color = 'red')
])

fig.update_layout(barmode='stack')
fig.update_layout(title_text='Daily cases of Covid-19 in India',yaxis=dict(title='Number of cases recorded' ),xaxis=dict(title='Date' ))

fig.show()
df = combinedDf.groupby('Date').sum().groupby('Date').cumsum()
cumDate= df.cumsum()
cumDate

#display
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=cumDate.index.tolist(), y=cumDate['Dead'],
    hoverinfo='x+y',
    mode='lines+markers',
     line_color='grey',
    stackgroup='one',
    name = "Dead"
))

fig.add_trace(go.Scatter(x=cumDate.index.tolist(), y=cumDate['Recovered'],
    hoverinfo='x+y',
    mode='lines+markers',
    line_color='green',
    stackgroup='one',
    name = "Recovered"
))
fig.add_trace(go.Scatter( x=cumDate.index.tolist(), y=cumDate['Confirmed'],
    hoverinfo='x+y',
    mode='lines+markers',
    line_color='red',
    stackgroup='one',
    name ="Confirmed"
))
fig.update_layout(title_text='Cummulative Trends of Covid-19 India',yaxis=dict(title='Number of cases' ),xaxis=dict(title='Date'))
fig.show()
import numpy as np
riDf = combinedDf.groupby('Date').sum()
recoveryRate = round((sum(combinedDf['Recovered'])+sum(combinedDf['Dead']))/sum(combinedDf['Confirmed']),2)

grratelist =[]
for i in range(len(riDf)-1):
    grratelist.append(round(riDf['Confirmed'][i+1]/riDf['Confirmed'][i],2))

rrratelist= []
for i in range(len(cumDate)):
    rrratelist.append(round(((cumDate['Recovered'][i]+cumDate['Dead'][i])/cumDate['Confirmed'][i]),2))

#display
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=cumDate.index.tolist()[:-1], y=rrratelist[:-1],
    hoverinfo='x+y',
    mode='lines+markers',
     line_color='green',
    stackgroup='one',
    name = "recovery"
))
fig.add_trace(go.Scatter(
    x=riDf.index.tolist()[:-1], y=grratelist,
    hoverinfo='x+y',
    mode='lines+markers',
     line_color='red',
    stackgroup='one',
    name = "growth"
))

print("Average Rate of growth : ",round(np.mean(rrratelist),2))
fig.update_layout(title_text='Daily Growth Rate and Recovery Rate of Covid-19 in India',yaxis=dict(title='Rate' ),xaxis=dict(title='Date' ))
fig.show()





