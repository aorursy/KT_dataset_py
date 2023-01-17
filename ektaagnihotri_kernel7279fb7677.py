import pandas as pd
import numpy as np
import datetime
import requests
import warnings

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import seaborn as sns
import squarify
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima_model import ARIMA
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot


import gc
import os
from pathlib import Path
import random
import sys

from tqdm.notebook import tqdm
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

from statsmodels.tsa.stattools import adfuller


from IPython.display import Image
warnings.filterwarnings('ignore')
%matplotlib inline

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly_express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.templates.default = "plotly_white"

path="/kaggle/input/covid19-in-india/"
def load_data(data):
    return pd.read_csv(path+data)
covid_19_india=pd.read_csv('../input/covid19-in-india/covid_19_india.csv', parse_dates=['Date'],dayfirst=True) # Reading Covid-19 Csv
print(covid_19_india.tail())
covid_19_state = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
covid_19_state['Date'] = pd.to_datetime(covid_19_state['Date'], dayfirst=True)
covid_19_state = covid_19_state.drop('Sno',axis=1)
age_group=load_data('AgeGroupDetails.csv') #Age Group Details Loading data using load_data func!!
ind_det=load_data('IndividualDetails.csv') #Individual details Loading data using load_data func!!

# Similary Loading other csv using load_data func

state_Test=load_data("StatewiseTestingDetails.csv")

bed=load_data('HospitalBedsIndia.csv') # Loading bed details

print(ind_det.current_status.unique())

#creating workable dataframe grouped by state and date. This is further used for plotting of trends per date
CS_covid_state = covid_19_india[['Date','State/UnionTerritory','Cured','Deaths','Confirmed']]
CS_covid_state = CS_covid_state.groupby(['Date','State/UnionTerritory'])[['Confirmed', 'Cured','Deaths']].sum().reset_index()
current_date=covid_19_india['Date'].max()
current_date
covid_19_india['Date'] = pd.to_datetime(covid_19_india['Date'], dayfirst=True)
df_covid = covid_19_india[['Date','State/UnionTerritory','Cured','Deaths','Confirmed']]
df_covid.rename(columns={'State/UnionTerritory':'States'}, inplace=True)
df_covid = df_covid.groupby(['Date'])[['Confirmed', 'Cured','Deaths']].sum().reset_index()
df_covid.tail(10)
covid_19_india = covid_19_india.groupby(['Date','State/UnionTerritory'])[['Confirmed', 'Cured','Deaths']].sum().reset_index()
covid_19_india.tail()

temp = covid_19_india[["Date","Confirmed","Deaths","Cured"]]
temp['Date'] = temp['Date'].apply(pd.to_datetime, dayfirst=True)

date_wise_data = temp.groupby(["Date"]).sum().reset_index()

total_data = date_wise_data.melt(id_vars="Date", value_vars=['Cured', 'Deaths', 'Confirmed'],
               var_name='Case', value_name='Count')

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
#temp.head()
cnf = '#393e46' # confirmed - grey
dth = '#ff2e63' # death - red
rec = '#21bf73' # recovered - cyan
act = '#fe9801'
fig = px.area(total_data, x="Date", y="Count", color='Case',title='Cases over time', color_discrete_sequence = [rec, dth, act])
fig.update_xaxes(rangeslider_visible=True)
fig.show()
df_covid['Active']=df_covid['Confirmed']-df_covid['Cured']-df_covid['Deaths']
df_covid['growth/day']=df_covid['Confirmed']-df_covid['Confirmed'].shift(1)
df_covid['cured/day']=df_covid['Cured']-df_covid['Cured'].shift(1)
df_covid['growth_ratio'] = round(df_covid['growth/day'] / df_covid['growth/day'].shift(1),2)
df_covid['recovery_ratio'] = round(df_covid['cured/day'] /df_covid['cured/day'].shift(1),2)
df_covid_melt = pd.melt(df_covid, id_vars=['Date'], value_vars=['Confirmed','Active','Cured','Deaths','cured/day','growth/day'])
import plotly.graph_objects as go
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
fig = px.line(df_covid_melt, x="Date", y="value", color='variable', title=f'All-India Cases with {current_date}')

#fig = px.line(df_covid, x='Date', y='Cured', title='Time Series with Rangeslider')

fig.update_xaxes(rangeslider_visible=True)
fig.show()
fig = px.line(df_covid_melt, x="Date", y="value", color='variable',
              title="All-India Cases Over Time (Log scale)",
             log_y=True)


fig.update_xaxes(rangeslider_visible=True)
fig.show()
df_covid['mortality'] = df_covid['Deaths'] / df_covid['Confirmed']

fig = px.line(df_covid, x="Date", y="mortality", 
              title="All-India Mortality Rate Over Time")
fig.show()
# Creating a custom table for better understanding of the data 
state_details = pd.pivot_table(covid_19_state, values=['Confirmed','Deaths','Cured'], index='State/UnionTerritory', aggfunc='max')
# Calculating the recovery rate which is Cured/Confirmed rounding to 2 digits
state_details['Recovery Rate'] = round(state_details['Cured'] / state_details['Confirmed'],2)
# Similarly, for Death Rate
state_details['Death Rate'] = round(state_details['Deaths'] /state_details['Confirmed'], 2)

state_details = state_details.sort_values(by='Confirmed', ascending= False).reset_index(level=0)

state_details.style.background_gradient(cmap='viridis', low=.5, high=0).highlight_null('red')
state_details.rename(columns={'State/UnionTerritory':'States'}, inplace=True)
import plotly.graph_objs as go
import plotly.offline as pyo # Setting Notebook to work Offline with Plotly
import plotly
pyo.init_notebook_mode()

# Acessing the values from state_details
x = state_details.States

confirmed_Cases = {
  'x': x,
  'y': state_details.Confirmed,# Created a trace variable to store confirmed cases as a bar per state wise ,similarly for Cured and Deaths
  'name': 'Confirmed',
  'type': 'bar',
    'marker': {
    'color': 'rgb(3, 166, 228)'
  }
};
confirmed_Cured = {
  'x': x,
  'y': state_details.Cured,
  'name': 'Cured',
  'type': 'bar',
    'marker': {
    'color': 'rgb(55, 151, 0)'
  }
};

confirmed_Deaths = {
  'x': x,
  'y': state_details.Deaths,
  'name': 'Deaths',
  'type': 'bar',
  'marker': {
    'color': 'rgb(216, 0, 0)'
  }
};

data = [confirmed_Cases, confirmed_Cured,confirmed_Deaths]; # A singleton row matrix to store the trace1,trace2,trace3

layout = {
  'xaxis': {'title': ' State-Data '},
  'barmode': 'relative',
  'title': 'Confirmed/Cured/Death Statewise Disturbution'
};

fig = go.Figure(data = data, layout = layout)#Plotting the bar plot form the above data 
pyo.iplot(fig)
total_confirmed_cases=covid_19_india[covid_19_india.Date==current_date]
max_confirmed_cases=total_confirmed_cases.sort_values(by="Confirmed",ascending=False)
top_states=max_confirmed_cases[0:5]
top_states.rename(columns={'State/UnionTerritory':'States'}, inplace=True)
top_states
top_states['Active'] = top_states['Confirmed'] - top_states['Cured'] - top_states['Deaths']
top_states_df = pd.melt(top_states, id_vars='States', value_vars=['Confirmed','Active', 'Cured','Deaths'])
fig = px.bar(top_states_df.iloc[::-1],
             x='value', y='States', color='variable', barmode='group',
             title=f'Confirmed/Cured/Deaths as on {current_date}', text='value', height=800, orientation='h')
fig.show()
top_states['recovery/day']=top_states['Cured']-top_states['Cured'].shift(1)
top_states['recovery/day'] = top_states['recovery/day'].replace(np.nan,'0.0')

top_states['new_case/day'] = top_states['Confirmed'] - top_states['Confirmed'].shift(1)
top_states['growth_ratio'] = top_states['new_case/day'] / top_states['new_case/day'].shift(1)

top_states['new_case/day'] = top_states['new_case/day'].replace(np.nan,'0.0')
top_states['growth_ratio'] = top_states['growth_ratio'].replace(np.nan,'0.0')

top_states

covid_19_state_ = covid_19_state.query('Date > "2020-01-01"')
covid_19_state_.head()

covid_19_state_['prev_confirmed'] = covid_19_state_.groupby('State/UnionTerritory')['Confirmed'].shift(1)
covid_19_state_['new_case'] = covid_19_state_['Confirmed'] - covid_19_state_['prev_confirmed']
covid_19_state_['new_case'].fillna(0, inplace=True)

covid_19_state_['prev_new_case'] = covid_19_state_.groupby('State/UnionTerritory')['new_case'].shift(1)
covid_19_state_['growth_factor'] = covid_19_state_['new_case'] / covid_19_state_['prev_new_case']
covid_19_state_['growth_factor'].fillna(0, inplace=True)
covid_19_state_['growth_factor'] = covid_19_state_['growth_factor'].replace(float('inf'),'0')

covid_19_state_['prev_cured'] = covid_19_state_.groupby('State/UnionTerritory')['Cured'].shift(1)
covid_19_state_['cured_case_per_day'] = covid_19_state_['Cured'] - covid_19_state_['prev_cured']

covid_19_state_['active'] = covid_19_state_['Confirmed'] - covid_19_state_['Cured'] - covid_19_state_['Deaths']

covid_19_state_Delhi = covid_19_state_[covid_19_state_['State/UnionTerritory']=='Delhi']
def state_wise_patients(name,df):
    data = df.loc[df['State/UnionTerritory']==name]
    df = data[['Confirmed','Cured','new_case','growth_factor','Date','State/UnionTerritory','active']]
    data = data.reset_index()
    data['Date']=pd.to_datetime(data['Date'],format = '%d/%m/%Y')
    data = data.sort_values(by=['Date'], ascending=True)
    return data
collection = {}
collection['Patients_in_Delhi'] = state_wise_patients('Delhi',covid_19_state_)
keys = list(collection.keys())
print(keys)
visible_True=[]
for i in range(len(keys)):
    visible_True.append(True)
def t2f(i):
    visible = []
    for a in range(len(keys)):
        if a == i:
            visible.append(True)
        else:
            visible.append(False)
    return visible
def create_buttons(keys,title):
    l=[dict(label = 'All',
                  method = 'update',
                  args = [{'visible': visible_True},
                          {'title': title+' India',
                           'showlegend':True}])]
    for i in range(len(keys)):
        l.append(dict(label = keys[i],
                  method = 'update',
                  args = [{'visible': t2f(i)}, # the index of True aligns with the indices of plot traces
                          {'title': title+keys[i].split('in')[1],
                           'showlegend':True}]))
    return l
# Plotting trend for state Delhi
fig = go.Figure()
keys = list(collection.keys())
for column in collection:
    fig.add_trace(
        go.Line(
            x = collection[column].Date,
            y = collection[column].growth_factor,
            name = 'Growth Factor'
        )
    )
    fig.add_trace(
        go.Line(
            x = collection[column].Date,
            y = collection[column].new_case/100,
            name = 'New Cases/100'
        )
    )
    fig.add_trace(
        go.Line(
            x = collection[column].Date,
            y = collection[column].active/1000,
            name = 'Active Cases/1000'
        )
    )
    fig.add_trace(
        go.Line(
            x = collection[column].Date,
            y = collection[column].Deaths/1000,
            name = 'Deaths/1000'
        )
    )
    fig.add_trace(
        go.Line(
            x = collection[column].Date,
            y = collection[column].cured_case_per_day/1000,
            name = 'Cured Cases per day/1000'
        )
    )
    fig.add_trace(
        go.Line(
            x = collection[column].Date,
            y = collection[column].Confirmed/1000,
            name = 'Confirmed Cases/1000'
        )
    )
   
fig.update_layout(updatemenus=[go.layout.Updatemenu( active=0,buttons=list(create_buttons(keys,'Confirmed Cases:')))])
fig.update_xaxes(rangeslider_visible=True)
fig.show()
#another way of plotting trend for Delhi
covid_19_Del_melt = pd.melt(covid_19_state_Delhi, id_vars=['Date'], value_vars=['Confirmed','active', 'Cured','Deaths','cured_case_per_day','growth_factor'])
target_date = covid_19_Del_melt['Date'].max()

fig = px.line(covid_19_Del_melt, x="Date", y="value", color='variable', 
              title=f'Trend Chart for Delhi as on {target_date}')

fig.update_xaxes(rangeslider_visible=True)
    
fig.show()
covid_19_state_Delhi['mortality'] = covid_19_state_Delhi['Deaths'] / covid_19_state_Delhi['Confirmed']

fig = px.line(covid_19_state_Delhi, x="Date", y="mortality", 
              title="Delhi Mortality Rate Over Time")
fig.show()
covid_19_state_Delhi.corr()

fig = go.Figure()

# Use the hovertext kw argument for hover text
fig = go.Figure(data=[go.Bar(x=age_group['AgeGroup'], y=age_group['TotalCases'])])
# Customize aspect
fig.update_traces(marker_color='rgb(3, 166, 228)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
fig.update_layout(title="Age wise Disturbution",yaxis_title="Total Number of cases",xaxis_title="Age Group")
fig.show()
ind_det.head()
px.histogram(ind_det, x='gender', color_discrete_sequence = ['indianred'], title='GenderWise  Distribution')

cp1=ind_det.copy()
cp1['current_status']=cp1['current_status'].replace(np.nan,'NaN')
cp2=cp1[cp1.gender == 'M'].groupby(['current_status']).count().reset_index()
cp3=cp1[cp1.gender == 'F'].groupby(['current_status']).count().reset_index()

cp2.rename(columns = {'id':'Male'}, inplace = True) 
cp3.rename(columns = {'id':'Female'}, inplace = True) 
cp4=pd.concat([cp2["current_status"],cp2["Male"] ,cp3["Female"]], axis=1).reset_index(drop=True, inplace=False)
cp4.style.background_gradient(cmap='terrain')
c4=cp4.set_index('current_status',inplace=True)  #reset the index and transpose the dataframe
c4=cp4.transpose().reset_index()
# prepare data frames
#df2014 = timesData[timesData.year == 2014].iloc[:3,:]
# import graph objects as "go"
import plotly.graph_objs as go
#from chart_studio.plotly import iplot
import plotly
from plotly.offline import iplot
# your code

x = c4.index


trace1 = {
  'x': x,
  'y': c4.Hospitalized,
  'name': 'Hospitalized',
  'type': 'bar'
};

trace2 = {
  'x': x,
  'y': c4.Deceased,
  'name': 'Deceased',
  'type': 'bar'
};

trace3 = {
  'x': x,
  'y': c4.Recovered,
  'name': 'Recovered',
  'type': 'bar'
};


data = [trace1, trace2,trace3];
layout = {
  'xaxis': {'title': 'Male-vs-Female'},
  'barmode': 'relative',
  'title': 'Gender-Wise-Disturbution of cases'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)
locations = ind_det.groupby(['detected_state', 'detected_district', 'detected_city'])['government_id'].count().reset_index()
locations['country'] = 'India'
fig = px.treemap(locations, path=["country", "detected_state", "detected_district", "detected_city"], values="government_id", height=700,
           title='State ---> District --> City', color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value+percent entry+percent root'
fig.show()
bed.rename(columns={'State/UT':'STUT'}, inplace=True)
bed.head()
bed["Total_bed"]=bed["NumPublicBeds_HMIS"]+bed["NumRuralBeds_NHP18"]+bed["NumUrbanBeds_NHP18"]
bed.tail()
df_bed=bed[:-1]
import plotly.express as px
#data_canada = px.data.gapminder().query("country == 'Canada'")
fig = px.bar(bed[:-1], x='STUT', y='Total_bed', text='Total_bed')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',title="Disturbution of Beds in Hospital")
fig.show()
import plotly.graph_objs as go
#from chart_studio.plotly import iplot
import plotly
from plotly.offline import iplot
# your code

x = bed.STUT[:-1]

trace1 = {
  'x': x,
  'y': bed.NumPrimaryHealthCenters_HMIS,
  'name': 'Public Health Center',
  'type': 'bar'
};
trace2 = {
  'x': x,
  'y': bed.NumCommunityHealthCenters_HMIS,
  'name': 'Community-hospitals',
  'type': 'bar'
};



trace3 = {
  'x': x,
  'y': bed.NumSubDistrictHospitals_HMIS,
  'name': 'sub-district',
  'type': 'bar'
};

trace4 = {
  'x': x,
  'y': bed.NumDistrictHospitals_HMIS,
  'name': 'District hospitals',
  'type': 'bar'
};


data = [trace1, trace2,trace3,trace4];
layout = {
  'xaxis': {'title': 'Statewise hospital Disturbution'},
  'barmode': 'relative',
  'title': 'Disturbution of Number of Hospitals'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)
covid_19_india.rename(columns={'State/UnionTerritory':'States'}, inplace=True)