import pandas as pd

import requests

from io import StringIO



from plotly.subplots import make_subplots

import plotly.graph_objects as go



from datetime import datetime, timedelta

from scipy.optimize import curve_fit

from numpy import asarray, exp, linspace





def f(day, day_turn, slope):

    return exp((day_turn-day)/slope)



def f_case(day, case, day_turn, slope, n=5):

    # total case function

    fval = f(day, day_turn, slope)

    return case/(1 + fval)**n



def df_case(day, case, day_turn, slope, n):

    # derivative of the total case function

    fval = f(day, day_turn, slope)

    return n * case/slope * fval / (1 + fval)**(n+1)



def f_active(day, case, day_turn, slope, n=5):

    return slope * df_case(day, case, day_turn, slope, n)
def countryFig(country, fig, row, col, annotations, province=None, showlegend=False, linestyle='solid',

              pcase=(1,1,5), pactive=(10,10,5)):

    

    if province==None:

        countryData = data.loc[data['Country/Region']==country]

        label = country

    else:

        countryData = data.loc[data['Country/Region']==country].loc[data['Province/State']==province]

        label = '{} / {}'.format(province, country)

    

    # take data

    dates     = countryData['Date']

    confirmed = countryData['Confirmed']

    recovered = countryData['Recovered']

    deaths    = countryData['Deaths']

    actives   = confirmed - recovered - deaths 

    

    # fit the data

    days_date = [datetime.strptime(di, '%Y-%m-%d') for di in dates]

    days = asarray([(di-days_date[0]).days for di in days_date])

    

    popt_case,   pcov_case   = curve_fit(f_case,   days, confirmed, p0 = pcase)

    popt_active, pcov_active = curve_fit(f_active, days, actives,   p0 = pactive)

    

    #print(popt_case)#,   pcov_case)

    #print(popt_active)#, pcov_active)

    

    days_extended_date = days_date + [days_date[-1] + di*timedelta(days=1) for di in days + 1]

    days_extended = asarray([(di-days_extended_date[0]).days for di in days_extended_date])

    

    fit_case   = f_case(days_extended, *popt_case)

    fit_active = f_active(days_extended, *popt_active)



    

    fig.add_trace(

    go.Bar(x=dates, y=confirmed,

               marker = go.Marker(color= 'rgb(255, 0, 0)'),

               name = "Total",

               showlegend=showlegend),        

    row=row, col=col)

    

    fig.add_trace(

    go.Bar(x=dates, y=actives,

               marker = go.Marker(color= 'rgb(0, 0, 255)'),

               name = "Active",

               showlegend=showlegend),

    row=row, col=col)



    fig.add_trace(

    go.Scatter(x=days_extended_date, y=fit_case,

               marker = go.Marker(color= 'rgb(255, 0, 0)'),

               line={'dash':'solid'},

               name = "Total - fit",

               showlegend=showlegend),

    row=row, col=col)

    

    fig.add_trace(

    go.Scatter(x=days_extended_date, y=fit_active,

               marker = go.Marker(color= 'rgb(0, 0, 255)'),

               line={'dash':'solid'},

               name = "Active - fit",

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
url = 'https://datahub.io/core/covid-19/r/time-series-19-covid-combined.csv'



headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:66.0) Gecko/20100101 Firefox/66.0"}

req = requests.get(url, headers=headers)

data_text = StringIO(req.text)



data = pd.read_csv(data_text)
fig = make_subplots(

    rows=2, cols=2,

    specs=[[{'type': 'scatter'}, {'type': 'scatter'}],

           [{'type': 'scatter'}, {'type': 'scatter'}]])



# adding surfaces to subplots.

annotations = []

countryFig('Turkey', fig, 1, 1, annotations, showlegend=True, linestyle='dot')  #0,1

countryFig('Iran', fig, 1, 2, annotations, linestyle='dot')                     #1,1   

countryFig('Germany', fig, 2, 1, annotations, linestyle='dot')                  #0,0

countryFig('China', fig, 2, 2, annotations, province='Hubei', linestyle='dot')  #1,0



fig.update_layout(

    title_text=r'COVID-19: Confirmed Total Cases vs Active Cases',

    autosize=False,

    height=900,

    width=900,

    #margin=dict(l=65, r=50, b=65, t=90),

    annotations = annotations

    )



fig.show()
fig = make_subplots(

    rows=2, cols=2,

    specs=[[{'type': 'scatter'}, {'type': 'scatter'}],

           [{'type': 'scatter'}, {'type': 'scatter'}]])



# adding surfaces to subplots.

annotations = []

countryFig('Turkey', fig, 1, 1, annotations, showlegend=True, linestyle='dot')  #0,1

countryFig('US', fig, 1, 2, annotations, linestyle='dot')                     #1,1   

countryFig('Spain', fig, 2, 1, annotations, linestyle='dot')                  #0,0

countryFig('Italy', fig, 2, 2, annotations, linestyle='dot')  #1,0



fig.update_layout(

    title_text=r'COVID-19: Confirmed Total Cases vs Active Cases',

    autosize=False,

    height=900,

    width=900,

    #margin=dict(l=65, r=50, b=65, t=90),

    annotations = annotations

    )



fig.show()
fig = make_subplots(

    rows=2, cols=2,

    specs=[[{'type': 'scatter'}, {'type': 'scatter'}],

           [{'type': 'scatter'}, {'type': 'scatter'}]])



# adding surfaces to subplots.

annotations = []

countryFig('Turkey', fig, 1, 1, annotations, showlegend=True, linestyle='dot')  #0,1

countryFig('Korea, South', fig, 1, 2, annotations, linestyle='dot')                     #1,1   

countryFig('Japan', fig, 2, 1, annotations, linestyle='dot')                  #0,0

countryFig('Israel', fig, 2, 2, annotations, linestyle='dot')  #1,0



fig.update_layout(

    title_text=r'COVID-19: Confirmed Total Cases vs Active Cases',

    autosize=False,

    height=900,

    width=900,

    #margin=dict(l=65, r=50, b=65, t=90),

    annotations = annotations

    )



fig.show()
fig = make_subplots(

    rows=2, cols=2,

    specs=[[{'type': 'scatter'}, {'type': 'scatter'}],

           [{'type': 'scatter'}, {'type': 'scatter'}]])



# adding surfaces to subplots.

annotations = []

countryFig('Turkey', fig, 1, 1, annotations, showlegend=True, linestyle='dot')  #0,1

countryFig('Lebanon', fig, 1, 2, annotations, linestyle='dot')                     #1,1   

countryFig('Egypt', fig, 2, 1, annotations, linestyle='dot')                  #0,0

countryFig('Pakistan', fig, 2, 2, annotations, linestyle='dot')  #1,0



fig.update_layout(

    title_text=r'COVID-19: Confirmed Total Cases vs Active Cases',

    autosize=False,

    height=900,

    width=900,

    #margin=dict(l=65, r=50, b=65, t=90),

    annotations = annotations

    )



fig.show()
N=7

countries = list(set(data['Country/Region']))

for i in [countries[x:x+N] for x in range(0, len(countries), N)]:

    print(i)