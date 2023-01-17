import pandas as pd

import numpy as np

import plotly.express as px

import matplotlib.pyplot as plt 

print('modules are imported')
dataset_url='https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'

df=pd.read_csv(dataset_url)
df.tail()
df.head()
df.shape
df=df[df['Confirmed']>0]
df.head()
df[df.Country=='Italy']
fig=px.choropleth(df,locations='Country',locationmode='country names',color='Confirmed'

                 ,animation_frame='Date')

fig.update_layout(title_text="Global spread of COVID-19")

fig.show()
fig=px.choropleth(df,locations="Country",locationmode='country names',color='Deaths',

                 animation_frame='Date')

fig.update_layout(title_text='Global Deaths because of COVID-19')

fig.show()
df_china=df[df.Country=='China']

df_china.head()
df_china=df_china[["Date",'Confirmed']]
df.head()
df_china["Infection Rate"]=df_china["Confirmed"].diff()
df_china.head()
px.line(df_china,x='Date',y=['Confirmed','Infection Rate'])
df_china["Infection Rate"].max()
df.head()
countries=list(df["Country"].unique())

max_infection_rates=[]

for c in countries:

    max_infected=df[df.Country==c].Confirmed.diff().max()

    max_infection_rates.append(max_infected)
df_MIR=pd.DataFrame()

df_MIR["Country"]=countries

df_MIR['Max Infection Rate']=max_infection_rates

df_MIR.head()
px.bar(df_MIR,x='Country',y='Max Infection Rate',color='Country',title='Global Max infection Rate',

      log_y=True)
italy_lockdown_start_date = '2020-03-09'

italy_lockdown_a_month_later = '2020-04-09'
df.head()
df_italy=df[df.Country=='Italy']
df_italy.head()
df_italy["Infection Rate"]=df_italy["Confirmed"].diff()

df_italy.head()
fig=px.line(df_italy,x='Date',y='Infection Rate',title="Before and After Lockdown")

fig.add_shape(dict( type ='line',x0=italy_lockdown_start_date,y0=0,

                   x1=italy_lockdown_start_date,y1=df_italy["Infection Rate"].max(),

                  line=dict(color='red',width=2)))

fig.show()

fig.add_annotation(dict(x=italy_lockdown_start_date,y=df_italy["Infection Rate"].max(),text='starting date of the lockdown'))
df_italy.head()
df_italy["Deaths Rate"]=df_italy["Deaths"].diff()
df_italy.head()
fig=px.line(df_italy,x='Date',y=["Infection Rate","Deaths Rate"])

fig.show()