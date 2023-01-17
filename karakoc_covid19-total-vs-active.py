import pandas as pd

import requests

from io import StringIO

from plotly.subplots import make_subplots

import plotly.graph_objects as go



url = 'https://datahub.io/core/covid-19/r/time-series-19-covid-combined.csv'



headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:66.0) Gecko/20100101 Firefox/66.0"}

req = requests.get(url, headers=headers)

data_text = StringIO(req.text)



data = pd.read_csv(data_text)
def countryFig(country, fig, row, col, annotations, province=None, showlegend=False, linestyle='solid'):

    

    if province==None:

        countryData = data.loc[data['Country/Region']==country]

        label = country

    else:

        countryData = data.loc[data['Country/Region']==country].loc[data['Province/State']==province]

        label = '{} / {}'.format(province, country)

    

    dates     = countryData['Date']

    confirmed = countryData['Confirmed']

    recovered = countryData['Recovered']

    deaths    = countryData['Deaths']

    actives   = confirmed - recovered - deaths 

    days = range(len(confirmed))

    

    fig.add_trace(

    go.Scatter(x=dates, y=confirmed,

               marker = go.Marker(color= 'rgb(0, 0, 255)'),

               line={'dash':linestyle},

               name = "Total",

               showlegend=showlegend),        

    row=row, col=col)

    

    fig.add_trace(

    go.Scatter(x=dates, y=actives,

               marker = go.Marker(color= 'rgb(255, 0, 0)'),

               line={'dash':linestyle},

               name = "Active",

               showlegend=showlegend),

    row=row, col=col)

    

    annotations += [

        dict(

            text=r'<b>{}</b>'.format(label),

            showarrow=False,

            xref="paper",

            yref="paper",

            x=col-1,

            y=2-row)

    ]
fig = make_subplots(

    rows=2, cols=2,

    specs=[[{'type': 'scatter'}, {'type': 'scatter'}],

           [{'type': 'scatter'}, {'type': 'scatter'}]])



# adding surfaces to subplots.

annotations = []

countryFig('Turkey', fig, 1, 1, annotations, showlegend=True)  #0,1

countryFig('Iran', fig, 1, 2, annotations)                     #1,1   

countryFig('Germany', fig, 2, 1, annotations)                  #0,0

countryFig('China', fig, 2, 2, annotations, province='Hubei')  #1,0



fig.update_layout(

    title_text=r'COVID-19: Confirmed Total Cases vs Active Cases',

    autosize=False,

    height=900,

    width=900,

    #margin=dict(l=65, r=50, b=65, t=90),

    annotations = annotations

    )



fig.show()
country, province = 'China', 'Hubei'



data.loc[data['Country/Region']==country].loc[data['Province/State']==province]