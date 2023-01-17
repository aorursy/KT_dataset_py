import numpy as np

import pandas as pd

import plotly.express as px

import plotly.graph_objs as go





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
PATH = "/kaggle/input/covid19-global-forecasting-week-1/"

train = pd.read_csv(PATH+"train.csv", parse_dates=['Date'])

test = pd.read_csv(PATH+"test.csv", parse_dates=['Date'])

# Taking daily updated and cleaned dataset from kaggle.com/imdevskp/corona-virus-report#covid_19_clean_complete.csv

cleaned = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv", parse_dates=['Date'])

cleaned.info()
cleaned.head()
train.rename(columns={'Province/State':'state','Country/Region':'country','ConfirmedCases': 'confirmed'}, inplace=True)

test.rename(columns={'Province/State':'state','Country/Region':'country','ConfirmedCases': 'confirmed'}, inplace=True)

cleaned.rename(columns={'Province/State':'state','Country/Region':'country','ConfirmedCases': 'confirmed'}, inplace=True)
cleaned.head()
cleaned['Date'].max()
print("Total locations : ", len(cleaned))

print("Total confirmations : ", cleaned['Confirmed'].sum())

print("Total deaths : ", cleaned['Deaths'].sum())

print("Total recoveries : ", cleaned['Recovered'].sum())
country_wise = cleaned.groupby('country')['Confirmed', 'Recovered', 'Deaths'].sum().reset_index()

print("Maximum confirmed cases found in : {}, with {} cases".format(country_wise[country_wise['Confirmed'].max()==country_wise['Confirmed']]['country'], country_wise['Confirmed'].max()))

country_wise
fig = px.choropleth(country_wise, 

                    locations="country", 

                    locationmode='country names', 

                    color="Confirmed", 

                    range_color=[0, 4500],

                    hover_name="country", 

                    color_continuous_scale=px.colors.sequential.Burg, 

                    title='Confirmed cases in various countries')

fig.show()
fig = px.choropleth(country_wise, 

                    locations="country", 

                    locationmode='country names', 

                    color="Deaths", 

                    range_color=[0, 1000],

                    hover_name="country", 

                    color_continuous_scale=px.colors.sequential.Burg, 

                    title='Death tolls in various countries')

fig.show()
fig = px.choropleth(country_wise, 

                    locations="country", 

                    locationmode='country names', 

                    color="Recovered", 

                    range_color=[0, 1000],

                    hover_name="country", 

                    color_continuous_scale=px.colors.sequential.Burg, 

                    title='Recovered cases in various countries')

fig.show()
date_wise = cleaned.groupby('Date')['Date','Confirmed','Deaths', 'Recovered'].sum().reset_index()

date_wise.head()
fig = px.line(date_wise, x="Date", y="Confirmed", 

              title="All confirmed cases over time")

fig.add_scatter(x=date_wise["Date"], y=date_wise["Deaths"], mode='lines', name='Deaths')

fig.add_scatter(x=date_wise["Date"], y=date_wise["Recovered"], mode='lines', name='Recovered')

fig.show()
top = country_wise.sort_values('Confirmed', ascending=False)[:10][::-1]



fig = go.Figure(data=[

    go.Bar(name='Confirmed', y=top['Confirmed'], x=top['country']),

    go.Bar(name='Deaths', y=top['Deaths'], x=top['country']),

    go.Bar(name='Recovered', y=top['Recovered'], x=top['country'])

])



fig.update_layout(barmode='group', title='Top 10 countries with most number of Confirmed cases')

fig.show()
top9 = top[top['country']!='China']



fig = go.Figure(data=[

    go.Bar(name='Confirmed', y=top9['Confirmed'], x=top9['country']),

    go.Bar(name='Deaths', y=top9['Deaths'], x=top9['country']),

    go.Bar(name='Recovered', y=top9['Recovered'], x=top9['country'])

])



fig.update_layout(barmode='group', title='Top 10 countries with most number of Confirmed cases')

fig.show()
fig = px.line(date_wise, x="Date", y="Confirmed", log_y=True,

              title="All confirmed cases over time")

fig.show()
fig = px.line(date_wise, x="Date", y="Recovered", log_y=True,

              title="All recovered patients over time")

fig.show()
fig = px.line(date_wise, x="Date", y="Deaths", log_y=True,

              title="All Dead cases over time")

fig.show()