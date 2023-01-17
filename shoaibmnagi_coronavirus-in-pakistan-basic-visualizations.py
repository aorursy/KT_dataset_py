import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df['ObservationDate'] = pd.to_datetime(df['ObservationDate'], format = '%m/%d/%Y')





df.head()

df.shape



df_pakistan = df[df['Country/Region'] == 'Pakistan']

df_pakistan.head()

df_pakistan.shape
len(df_pakistan['ObservationDate'].unique())



df_pakistan['Province/State'] = df_pakistan['Province/State'].fillna('National') # to signify that this number represents that national number of cases



df_pakistan.isna().sum() #perfect, no missing values nonw
df_pak_national = df_pakistan.groupby('ObservationDate').sum() #beyond 9th June, we only have provincial totals, hence we need to sum for the national value

df_pak_national.tail()

fig = px.line(df_pak_national, y = 'Confirmed', title = 'Timeline of Confirmed Covid-19 Cases in Pakistan')



fig.update_xaxes(

    rangeslider_visible=True,

    rangeselector=dict(

        buttons=list([

            dict(count=1, label="1m", step="month", stepmode="backward"),

            dict(count=3, label="3m", step="month", stepmode="backward"),

            dict(step = 'all')

        ])

    )

)

fig.show()
fig = px.line(df_pakistan, x = 'ObservationDate', y = 'Confirmed', color = 'Province/State', title = 'Provincial Timeline of Confirmed Covid-19 Cases in Pakistan since 10th June')

fig.update_layout(xaxis_range=['2020-06-10','2020-07-08'])



fig.show()
fig = px.line(df_pak_national, y = 'Deaths', title = 'Timeline of Confirmed Covid-19 Fatalities in Pakistan')



fig.update_xaxes(

    rangeslider_visible=True,

    rangeselector=dict(

        buttons=list([

            dict(count=1, label="1m", step="month", stepmode="backward"),

            dict(count=3, label="3m", step="month", stepmode="backward"),

            dict(step = 'all')

        ])

    )

)

fig.show()
fig = px.line(df_pakistan, x = 'ObservationDate', y = 'Deaths', color = 'Province/State', title = 'Provincial Timeline of Confirmed Covid-19 Fatalities in Pakistan since 10th June')

fig.update_layout(xaxis_range=['2020-06-10','2020-07-08'])



fig.show()
fig = px.line(df_pak_national, y = 'Recovered', title = 'Timeline of Confirmed Covid-19 Recoveries in Pakistan')



fig.update_xaxes(

    rangeslider_visible=True,

    rangeselector=dict(

        buttons=list([

            dict(count=1, label="1m", step="month", stepmode="backward"),

            dict(count=3, label="3m", step="month", stepmode="backward"),

            dict(step = 'all')

        ])

    )

)

fig.show()
fig = px.line(df_pakistan, x = 'ObservationDate', y = 'Recovered', color = 'Province/State', title = 'Provincial Timeline of Confirmed Covid-19 Recoveries in Pakistan since 10th June')

fig.update_layout(xaxis_range=['2020-06-10','2020-07-08'])



fig.show()
df_pak_national['Death_Rate'] = df_pak_national['Deaths'] / df_pak_national['Confirmed'] * 100

df_pak_national['Recovery_Rate'] = df_pak_national['Recovered'] / df_pak_national['Confirmed'] * 100
fig = px.line(df_pak_national, y = 'Death_Rate', title = 'Timeline of Covid-19 Fatality Rates in Pakistan')



fig.update_xaxes(

    rangeslider_visible=True,

    rangeselector=dict(

        buttons=list([

            dict(count=1, label="1m", step="month", stepmode="backward"),

            dict(count=3, label="3m", step="month", stepmode="backward"),

            dict(step = 'all')

        ])

    )

)

fig.show()
fig = px.line(df_pak_national, y = 'Recovery_Rate', title = 'Timeline of Confirmed Covid-19 Recovery Rates in Pakistan')



fig.update_xaxes(

    rangeslider_visible=True,

    rangeselector=dict(

        buttons=list([

            dict(count=1, label="1m", step="month", stepmode="backward"),

            dict(count=3, label="3m", step="month", stepmode="backward"),

            dict(step = 'all')

        ])

    )

)

fig.show()
df_pakistan['Death_Rate'] = df_pakistan['Deaths'] / df_pakistan['Confirmed'] * 100

df_pakistan['Recovery_Rate'] = df_pakistan['Recovered'] / df_pakistan['Confirmed'] * 100



fig = px.line(df_pakistan, x = 'ObservationDate', y = 'Death_Rate', color = 'Province/State', title = 'Provincial Timeline of Covid-19 Fatality Rates in Pakistan since 10th June')

fig.update_layout(xaxis_range=['2020-06-10','2020-07-08'])



fig.show()
fig = px.line(df_pakistan, x = 'ObservationDate', y = 'Recovery_Rate', color = 'Province/State', title = 'Provincial Timeline of Covid-19 Recovery Rates in Pakistan since 10th June')

fig.update_layout(xaxis_range=['2020-06-10','2020-07-08'])



fig.show()
df_pak_national["WeekOfYear"] = df_pak_national.index.weekofyear



week = []

pak_weekly_cases = []

pak_weekly_recoveries = []

pak_weekly_deaths = []



w = 1



for i in list(df_pak_national["WeekOfYear"].unique()):

    pak_weekly_cases.append(df_pak_national[df_pak_national["WeekOfYear"] == i]["Confirmed"].iloc[-1])

    pak_weekly_recoveries.append(df_pak_national[df_pak_national["WeekOfYear"] == i]["Recovered"].iloc[-1])

    pak_weekly_deaths.append(df_pak_national[df_pak_national["WeekOfYear"] == i]["Deaths"].iloc[-1])

    week.append(w)

    w = w + 1

    

fig = go.Figure()

fig.add_trace(go.Scatter(x = week, y = pak_weekly_cases,

                         mode = 'lines+markers',

                         name = 'Weekly Tally of Confirmed Cases'))

fig.add_trace(go.Scatter(x = week, y = pak_weekly_recoveries,

                         mode = 'lines+markers',

                         name = 'Weekly Tally of Recoveries'))

fig.add_trace(go.Scatter(x = week, y = pak_weekly_deaths,

                         mode = 'lines+markers',

                         name = 'Weekly Tally of Deaths'))

fig.update_layout(title = "Weekly Tally of Cases, Recoveries and Deaths in Pakistan",

                  xaxis_title = "Week Number of 2020",yaxis_title = "Total Number of Cases",

                  legend = dict(x = 0,y = 1,traceorder = "normal"))

fig.show()
from statsmodels.tsa.holtwinters import SimpleExpSmoothing



model = SimpleExpSmoothing(df_national_pak)

model_fit = model.fit()



yhat = model_fit.predict(len(data), len(data))
