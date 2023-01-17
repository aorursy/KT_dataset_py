import os

import sys

# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

import pycountry

import plotly.express as px

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot



import pandas as pd

import numpy as np

!pip install pywaffle

from pywaffle import Waffle



py.init_notebook_mode(connected=True)

import folium 

from folium import plugins

plt.style.use("fivethirtyeight")# for pretty graphs



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

#plt.rcParams['image.cmap'] = 'viridis'



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Disable warnings 

import warnings

warnings.filterwarnings('ignore')


df= pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

df.head()
df.drop(['Sno'],axis=1,inplace=True)
df['Total cases'] = df['Deaths'] + df['Cured'] + df['Confirmed']

df['Active cases'] = df['Total cases'] - (df['Cured'] + df['Deaths'])

print("Total Cases:",df['Total cases'].sum())

print("Active Cases:",df['Active cases'].sum())

print('Total number of Cured/Discharged/Migrated COVID 2019 cases across India:', df['Cured'].sum())

print('Total number of Deaths due to COVID 2019  across India:', df['Deaths'].sum())

print('Total number of States/UTs affected:', len(df['State/UnionTerritory']))
#https://www.kaggle.com/nxrprime/styling-data-frames-covid-19-vs-conferences

def highlight_max(s):

    is_max = s == s.max()

    return ['background-color: pink' if v else '' for v in is_max]



#df.style.apply(highlight_max,subset=['Total Confirmed cases (Indian National)', 'Total Confirmed cases ( Foreign National )'])

df.style.apply(highlight_max,subset=['Cured', 'Deaths','Total cases','Active cases'])
df['ConfirmedIndianNational'][400]
indian = df['ConfirmedIndianNational'].sum()

foreign = df['ConfirmedForeignNational'].sum()

x = df.groupby('State/UnionTerritory')['Active cases'].sum().sort_values(ascending=False).to_frame()

x.style.background_gradient(cmap='Reds')
fig = px.bar(df.sort_values('Active cases', ascending=False).sort_values('Active cases', ascending=True), 

             x="Active cases", y="State/UnionTerritory", 

             title='Total Active Cases', 

             text='Active cases', 

             orientation='h', 

             width=1000, height=700, range_x = [0, max(df['Active cases'])])

fig.update_traces(marker_color='#46cdcf', opacity=0.8, textposition='inside')



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
f, ax = plt.subplots(figsize=(12, 8))

data = df[['State/UnionTerritory','Total cases','Cured','Deaths']]

data.sort_values('Total cases',ascending=False,inplace=True)

sns.set_color_codes("pastel")

sns.barplot(x="Total cases", y="State/UnionTerritory", data=data,

            label="Total", color="r")



sns.set_color_codes("muted")

sns.barplot(x="Cured", y="State/UnionTerritory", data=data,

            label="Recovered", color="g")





# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 35), ylabel="",

       xlabel="Cases")

sns.despine(left=True, bottom=True)

# Rise in COVID-19 cases in India

fig = go.Figure()

fig.add_trace(go.Scatter(x=df['Date'], y=df['Total cases'],

                    mode='lines+markers',name='Total cases'))



fig.add_trace(go.Scatter(x=df['Date'], y=df['Cured'], 

                mode='lines',name='Recovered'))

fig.add_trace(go.Scatter(x=df['Date'], y=df['Active cases'], 

                mode='lines',name='Active'))

fig.add_trace(go.Scatter(x=df['Date'], y=df['Deaths'], 

                mode='lines',name='Deaths'))

        

    

fig.update_layout(title_text='Trend of Coronavirus Cases in India(Cumulative cases)',plot_bgcolor='rgb(250, 242, 242)')



fig.show()





# New COVID-19 cases reported daily in India



import plotly.express as px

fig = px.bar(df, x="Date", y="Total cases", barmode='group',

             height=400)

fig.update_layout(title_text='New Coronavirus Cases in India per day',plot_bgcolor='rgb(250, 242, 242)')



fig.show()