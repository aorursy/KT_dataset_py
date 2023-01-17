import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

import plotly
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df[["Confirmed","Deaths","Recovered"]] =df[["Confirmed","Deaths","Recovered"]].astype(int)

df['Country/Region'] = df['Country/Region'].replace('Mainland China', 'China')

df['Active_case']=df['Confirmed']-df['Deaths']-df['Recovered']

df.head()
df_world = df.groupby(["ObservationDate"])[["Confirmed","Active_case","Recovered","Deaths"]].sum().reset_index()

df_world.head()
df_world[-5:-1]
fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=df_world['ObservationDate'], y=df_world['Confirmed'],

                    mode='lines',

                    name='Confirmed cases'))

fig2.add_trace(go.Scatter(x=df_world['ObservationDate'], y=df_world['Active_case'],

                    mode='lines',

                    name='Active cases'))

fig2.add_trace(go.Scatter(x=df_world['ObservationDate'], y=df_world['Deaths'],name='Deaths',

                                   marker_color='black',mode='lines' ))

fig2.add_trace(go.Scatter(x=df_world['ObservationDate'], y=df_world['Recovered'],mode='lines',

                    name='Recovered cases',marker_color='green'))

fig2.update_layout(

    title='Evolution of cases over time',

        template='plotly_white')

fig2.show()
df_country = df.groupby(['Country/Region', 'ObservationDate']).sum().reset_index().sort_values('ObservationDate', ascending=False)

df_country = df_country.drop_duplicates(subset = ['Country/Region'])

df_country = df_country[df_country['Active_case']>0]

df_country.head()
fig = px.choropleth(df_country, locations=df_country['Country/Region'],

                    color=df_country['Active_case'],locationmode='country names', 

                    hover_name=df_country['Country/Region'])

fig.update_layout(

    title='Active Cases In Each Country',

)

fig.show()
df = df.rename(columns={'Country/Region':'Country'})

df = df.rename(columns={'ObservationDate':'Date'})

df_countrydate = df[df['Confirmed']>0]

df_countrydate = df_countrydate.groupby(['Date','Country']).sum().reset_index()

df_countrydate

# Creating the visualization

fig = px.choropleth(df_countrydate, 

                    locations="Country", 

                    locationmode = "country names",

                    color="Confirmed", 

                    hover_name="Country", 

                    animation_frame="Date"

                   )

fig.update_layout(

    title_text = 'Global Spread of Coronavirus',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

fig.show()
df_taiwan = df[(df['Country'] == 'Taiwan') ].reset_index(drop=True)

df_taiwan.head()
fig3 = go.Figure()

fig3.add_trace(go.Scatter(x=df_taiwan['Date'], y=df_taiwan['Confirmed'],

                    mode='lines',

                    name='Confirmed cases'))

fig3.add_trace(go.Scatter(x=df_taiwan['Date'], y=df_taiwan['Active_case'],

                    mode='lines',

                    name='Active cases'))

fig3.add_trace(go.Scatter(x=df_taiwan['Date'], y=df_taiwan['Deaths'],name='Deaths',

                                   marker_color='black',mode='lines',line=dict( dash='dot') ))

fig3.add_trace(go.Scatter(x=df_taiwan['Date'], y=df_taiwan['Recovered'],mode='lines',

                    name='Recovered cases',marker_color='green',line=dict( dash='dot')))

fig3.update_layout(

    title='Evolution of cases over time in Taiwan',

        template='plotly_white')

fig3.show()