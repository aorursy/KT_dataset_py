import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
covid_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",parse_dates=["ObservationDate"])
print(covid_data.head())
covid_data=covid_data.drop(['SNo','Last Update'], axis=1)
print(covid_data)
print(covid_data.describe())
covid_data.rename({'ObservationDate':'Date','Country/Region':'Country', 'Province/State':'State', 'Confirmed':'Confirmed Cases'}, axis=1, inplace=True)
print(covid_data.head())
covid_data_country=covid_data.groupby(['Date','Country']).sum().reset_index()
print(covid_data_country)
india_cases=covid_data[covid_data['Country']=='India']
india_cases.drop(['State'],axis=1)
print(india_cases)
india_states = india_cases[india_cases['State'].notna()]
india_states=india_states.drop(['Country'], axis=1)
print(india_states)
#Cases by states
covid_data_india_states=india_states.groupby(['State']).sum().reset_index()
print(covid_data_india_states)
covid_data['State'].value_counts().sort_values(ascending=True)
print(covid_data['State'].value_counts().sort_values(ascending=True))
covid_data['Country'].value_counts().sort_values(ascending=True)
print(covid_data['Country'].value_counts().sort_values(ascending=True))
covid_data['Date'].value_counts().sort_values(ascending=True)
print(covid_data['Date'].value_counts().sort_values(ascending=True))
dates = list(covid_data['Date'].dt.day)
india_cases.plot(x='Date',y='Confirmed Cases')
#After looking at the graph one can actually see the variation post june mid, So we're gonna check the graph further

june_cases=india_cases[(india_cases['Date'] > '2020-06-09') & (india_cases['Date'] <= '2020-06-27')]
june_cases.reset_index()
print(june_cases['Confirmed Cases'].value_counts())
june_dates=list(june_cases['Date'].dt.day)
june_cases.plot(x='Date',y='Confirmed Cases')
grouped=june_cases.groupby('State').sum()
print(grouped)
grouped.plot(kind='bar',x=None,y=['Confirmed Cases','Deaths','Recovered'])
