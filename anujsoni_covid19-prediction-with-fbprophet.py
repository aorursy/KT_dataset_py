import pandas as pd

import numpy as np

import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt

from fbprophet import Prophet
dataset_url="https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv"

df=pd.read_csv(dataset_url)

df = df[df['Confirmed']>0]

df.head()
fig=px.choropleth(df,locations='Country',locationmode='country names',color='Confirmed'

                 ,animation_frame='Date')

fig.update_layout(title_text='Global spread of COVID-19')

fig.show()
fig = px.choropleth(df,locations="Country",locationmode='country names',color='Deaths',animation_frame='Date')

fig.update_layout(title_text='Global Deaths of COVID-19')

fig.show()
fig = px.choropleth(df,locations="Country",locationmode='country names',color='Recovered',animation_frame='Date')

fig.update_layout(title_text='Global Recovered of COVID-19')

fig.show()
def visualizer(df,lockdown_start_date,lockdown_end_date,cases):

    print("VISUALIZER")

    print()

    print(df.head())

    print()

    print(df.tail())

    st = ""

    if cases == "Confirmed":

        st = "Infection Rate"

    if cases == "Deaths":

        st = "Death Rate"

    if cases == "Recovered":

        st = "Recovered Rate"

    df[st] = df[cases].diff()

    df = df[df[st]>0]

    

    fig=px.line(df,x="Date",y=st)

    fig.add_shape(dict(type="line",x0=lockdown_start_date,y0=0,x1=lockdown_start_date,y1=df[st].max(),line=dict(color="red",width=2)))

    fig.add_annotation(dict(x=lockdown_start_date,y=df[st].max(),text='starting date of the lockdown'))

    fig.add_shape(dict(type='line',x0=lockdown_end_date,y0=0,x1=lockdown_end_date,y1=df[st].max(),line=dict(color="red",width=2)))

    fig.add_annotation(dict(x=lockdown_end_date,y=df[st].max(),text="lockdown end date"))

    fig.show()

    

    print("Maximum "+st+" till now is ",df[st].max())

    df = df.dropna()

    df = df.drop(st,axis=1)

    df.groupby("Country").sum().plot.barh()

    
def predictor(df,cases):

    print("PREDICTOR")

    df = df.sort_values('Date')

    df = df[df[cases]>0]

    prophet_df = df[['Date',cases]]

    print(prophet_df.head())

    prophet_df = prophet_df.rename(columns={'Date':'ds',cases:'y'})

    m = Prophet()

    m.fit(prophet_df)

    future = m.make_future_dataframe(periods=365)

    forecast = m.predict(future)

    m.plot(forecast,xlabel='Date',ylabel=cases)

    forecast.to_csv('/home/anujsoni/output.csv')
input_df = input("Enter country name")

lsd = input("lockdown start date")

led = input("lockdown end date")

cases = input("Enter case(Confirmed,Deaths or Recovered)")

visualizer(df[df["Country"]==input_df],lsd,led,cases)

predictor(df[df["Country"]==input_df],cases)