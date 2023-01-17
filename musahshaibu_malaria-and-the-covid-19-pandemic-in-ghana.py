import numpy as np

import pandas as pd 

import plotly.express as px

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import matplotlib.dates as mdates

import plotly.express as px

from fbprophet import Prophet

from collections import namedtuple

import pycountry

import seaborn as sns
c9 = pd.read_csv("../input/covid19-ghana/CoVidGH.csv")

md = pd.read_csv("../input/malariadata-ghana/MalarialDeathByAge.csv")

df = pd.read_csv('../input/ghana-dataset/select-malaria-indicators_subnational_gha.csv')



c9.head()
c9.groupby("country")[['cases', 'deaths', 'Recovery']].sum().reset_index()
c9.iloc[:,:-1].corr().style.background_gradient(cmap='Reds')
md.head()
md.groupby("Entity")[['UnderFiveYearsDeath', 'FiveFouteenYearsOldDeaths', 'SeventyYearsOldDeaths']].sum().reset_index()
symptoms={'CoVid19 Symptom':['Fever',

        'Dry cough',

        'Fatigue',

        'Sputum production',

        'Shortness of breath',

        'Muscle pain',

        'Sore throat',

        'Headache',

        'Chills',

        'Nausea or vomiting',

        'Nasal congestion',

        'Conjunctival congestion'],

        

        #Malaria Symptoms

        'Malaria Symptom':['Fever',

        'a high temperature of 38C or above',

        'feeling hot and shivery',

        'headaches',

        'Shortness of breath',

        'Muscle pain',

        'Sore throat',

        'vomiting',

        'Nasal congestion',

        'Diarrhoea',

        'Haemoptysis',

        'Conjunctival congestion'],'percentage':[87.9,76.7,68.1,59.4,18.6,25.8,20.9,11.4,5.0,4.8,3.7,0.9]}

         

   

symptoms=pd.DataFrame(data=symptoms,index=range(12))

symptoms



fig = px.bar(symptoms[['CoVid19 Symptom', 'percentage']].sort_values('percentage', ascending=False), 

             y="percentage", x="CoVid19 Symptom", color='CoVid19 Symptom', 

             log_y=True, template='ggplot2', title='Symptom of  Coronavirus')

fig.show()
fig = px.bar(symptoms[['Malaria Symptom', 'percentage']].sort_values('percentage', ascending=False), 

             y="percentage", x="Malaria Symptom", color='Malaria Symptom', 

             log_y=True, template='ggplot2', title='Symptom of  Malaria')

fig.show()
confirmed_total_date = c9.groupby(['dateRep']).agg({'cases':['sum']})

malaria_infection_rate = md.groupby(['Year']).agg({'population with malaria':['sum']})

total_date = confirmed_total_date.join(malaria_infection_rate)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

total_date.plot(ax=ax1)

ax1.set_title("Covid19 Confirmed Cases", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

malaria_infection_rate.plot(ax=ax2, color='orange')

ax2.set_title("Malaria infection Cases", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)


confirmed_total_date = c9.groupby(['dateRep']).agg({'deaths':['sum']})

malaria_infection_rate = md.groupby(['Year']).agg({'DeathsMalariaThousand':['sum']})

total_date = confirmed_total_date.join(malaria_infection_rate)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

total_date.plot(ax=ax1)

ax1.set_title("Covid19 Death", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

malaria_infection_rate.plot(ax=ax2, color='orange')

ax2.set_title("Malaria Death", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
confirmed = c9.groupby('dateRep').sum()['cases'].reset_index()

deaths = c9.groupby('dateRep').sum()['deaths'].reset_index()

recovered = c9.groupby('dateRep').sum()['Recovery'].reset_index()



fig = go.Figure()

fig.add_trace(go.Scatter(x=confirmed['dateRep'], 

                         y=confirmed['cases'],

                         mode='lines+markers',

                         name='Confirmed Cases',

                         line=dict(color='blue', width=2)

                        ))

fig.add_trace(go.Scatter(x=deaths['dateRep'], 

                         y=deaths['deaths'],

                         mode='lines+markers',

                         name='Deaths',

                         line=dict(color='Red', width=2)

                        ))

fig.add_trace(go.Scatter(x=recovered['dateRep'], 

                         y=recovered['Recovery'],

                         mode='lines+markers',

                         name='Recovered',

                         line=dict(color='Green', width=2)

                        ))

fig.update_layout(

    title='Ghana Corona Virus Cases - Confirmed, Deaths, Recovered (Line Chart)',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Number of Cases',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)

fig.show()
deaths_5 = md.groupby('Year').sum()['UnderFiveYearsDeath'].reset_index()

deaths_15 = md.groupby('Year').sum()['FifteenFourtyNineYearsOldDeaths'].reset_index()

deaths_70 = md.groupby('Year').sum()['SeventyYearsOldDeaths'].reset_index()



fig = go.Figure()

fig.add_trace(go.Scatter(x=deaths_5['Year'], 

                         y=deaths_5['UnderFiveYearsDeath'],

                         mode='lines+markers',

                         name='Under 5s Death',

                         line=dict(color='blue', width=2)

                        ))

fig.add_trace(go.Scatter(x=deaths_15['Year'], 

                         y=deaths_15['FifteenFourtyNineYearsOldDeaths'],

                         mode='lines+markers',

                         name='15-49 Deaths',

                         line=dict(color='Red', width=2)

                        ))

fig.add_trace(go.Scatter(x=deaths_70['Year'], 

                         y=deaths_70['SeventyYearsOldDeaths'],

                         mode='lines+markers',

                         name='Above 70s Deaths',

                         line=dict(color='Green', width=2)

                        ))

fig.update_layout(

    title='Malaria Death in Ghana (Line Chart)',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Number of Deaths',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)

fig.show()
confirmed = c9.groupby('dateRep').sum()['cases'].reset_index()

deaths = c9.groupby('dateRep').sum()['deaths'].reset_index()

recovered = c9.groupby('dateRep').sum()['Recovery'].reset_index()

malcas = md.groupby('Year').sum()['population with malaria'].reset_index()
confirmed.columns = ['ds','y']

confirmed['ds'] = pd.to_datetime(confirmed['ds'])
confirmed.head()
m = Prophet(interval_width=0.95)

m.fit(confirmed)

future = m.make_future_dataframe(periods=7)

future_confirmed = future.copy() # for non-baseline predictions later on

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
confirmed_forecast_plot = m.plot(forecast)
forecast_components = m.plot_components(forecast)
deaths.columns = ['ds','y']

deaths['ds'] = pd.to_datetime(deaths['ds'])
m = Prophet(interval_width=0.95)

m.fit(deaths)

future = m.make_future_dataframe(periods=7)

future_deaths = future.copy() # for non-baseline predictions later on

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
deaths_forecast_plot = m.plot(forecast)
forecast_components = m.plot_components(forecast)
recovered.columns = ['ds','y']

recovered['ds'] = pd.to_datetime(recovered['ds'])
m = Prophet(interval_width=0.95)

m.fit(recovered)

future = m.make_future_dataframe(periods=7)

future_recovered = future.copy() # for non-baseline predictions later on

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
recovered_forecast_plot = m.plot(forecast)
forecast_components = m.plot_components(forecast)
malcas.columns = ['ds','y']

malcas['ds'] = pd.to_datetime(malcas['ds'])
m = Prophet(interval_width=0.95)

m.fit(malcas)

future = m.make_future_dataframe(periods=7)

future_malcas = future.copy() # for non-baseline predictions later on

future.tail()
malcas_forecast_plot = m.plot(forecast)
fig = px.bar(df[['Location', 'Value']].sort_values('Value', ascending=False), 

             y="Value", x="Location", color='Location', 

             log_y=True, template='ggplot2', title='Households with at least one insecticide-treated mosquito net (ITN)')

fig.show()