import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

import plotly

import plotly.offline as pyo

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

pyo.init_notebook_mode()
df= pd.read_csv('../input/india-covid19-datajanuary-to-september/India COVID-19.csv')

df.head()
df.isna().any()
df.drop('Unnamed: 0',axis=1,inplace=True)
df.info()
df['Date']=pd.to_datetime(df['Date'])
fig1=px.area(df,y=df['Confirmed'],x=df['Date'],labels={'Confirmed':'Confirmed cases'},color_discrete_sequence=['orange'])

fig1.update_layout(title='Rise in confirmed cases',title_x=0.5,template='plotly_dark')

fig1.show()
df['Month']=df['Date'].dt.month
df['Month'].replace({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep'},inplace=True)

df_month=df.groupby('Month')['Confirmed'].max().reset_index().sort_values(by='Confirmed')

df_month['Actual cases']=df_month['Confirmed'].diff(1)

df_month['Actual cases'].fillna(1,inplace=True)
fig2=px.bar(df_month,x='Month',y='Actual cases',color='Month',labels={'Actual cases':'Confirmed cases'})

fig2.update_layout(template='plotly_dark',title='Monthly case loads since first case',title_x=0.5)

fig2.show()
df['Day']=df['Date'].dt.dayofweek

df['Day']=df['Day'].replace({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
df_day=df.groupby('Day')['Confirmed'].max().reset_index().sort_values(by='Confirmed')
df_day['Actual confirmed']=df_day['Confirmed'].diff(1)
df_day['Actual confirmed'].fillna(75083,inplace=True)

df_day.replace({0:89000},inplace=True)
fig3=px.line(df_day,x='Day',y='Actual confirmed',labels={'Confirmed':'Confirmed cases'})

fig3.update_layout(title='Day wise case load',title_x=0.5,template='plotly_dark')

fig3.update_traces(line_color='#AAF0D1')

fig3.show()
fig4=px.area(df,y=df['Recovered'],x=df['Date'],labels={'Confirmed':'Confirmed cases'},color_discrete_sequence=['blue'])

fig4.update_layout(title='Rise in recovered cases',title_x=0.5,template='plotly_dark')

fig4.show()
fig5=px.area(df,y=df['Deaths'],x=df['Date'],labels={'Confirmed':'Confirmed cases'},color_discrete_sequence=['red'])

fig5.update_layout(title='Rise in fatal cases',title_x=0.5,template='plotly_dark')

fig5.show()
fig6=go.Figure()



fig6.add_trace(go.Scatter(name='Confirmed cases',x=df['Date'], y=df['Confirmed'], fill='tozeroy',

                    mode='none',fillcolor='blue' 

                    ))



fig6.add_trace(go.Scatter(name='Recovered cases',x=df['Date'], y=df['Recovered'], fill='tozeroy',

                    mode='none',fillcolor='green'

                    ))



fig6.add_trace(go.Scatter(name='Deaths',x=df['Date'], y=df['Deaths'], fill='tozeroy',

                    mode='none',fillcolor='red'

                    ))



fig6.update_layout(title='COVID-19 spread in India',title_x=0.5,template='plotly_dark')

fig6.show()