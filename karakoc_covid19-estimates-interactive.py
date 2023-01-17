from ipywidgets import Dropdown, Layout, GridspecLayout, Output, Button



import pandas as pd

import requests

from io import StringIO



from plotly.subplots import make_subplots

import plotly.graph_objects as go



from datetime import datetime, timedelta

from scipy.optimize import curve_fit

from numpy import asarray, exp, linspace
def countryFig(country, province, fig, row, col, annotations, showlegend=False, linestyle='solid',

              pcase=(10,10,5.), pactive=(10,10,5.)):

    

    countryData = data.loc[data['Country/Region']==country].loc[data['Province/State'].astype(str)==province]

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

    

    days_extended_date = days_date + [days_date[-1] + di*timedelta(days=1) for di in days + 1]

    days_extended_date = days_extended_date + [days_extended_date[-1] + di*timedelta(days=1) for di in days + 1]

    days_extended = asarray([(di-days_extended_date[0]).days for di in days_extended_date])

    

    fit_case   = f_case(days_extended, *popt_case)

    fit_active = f_active(days_extended, *popt_active)



    

    fig.add_trace(

    go.Bar(x=dates, y=confirmed,

               marker = go.bar.Marker(color= 'rgb(255, 0, 0)'),

               name = "Total",

               showlegend=showlegend),        

    row=row, col=col)

    

    fig.add_trace(

    go.Bar(x=dates, y=actives,

               marker = go.bar.Marker(color= 'rgb(0, 0, 255)'),

               name = "Active",

               showlegend=showlegend),

    row=row, col=col)

    

    fig.add_trace(

    go.Scatter(x=days_extended_date, y=fit_case,

               marker = go.scatter.Marker(color= 'rgb(255, 0, 0)'),

               line={'dash':'solid', 'width':4},

               name = "Total - fit",

               showlegend=showlegend),

    row=row, col=col)

    

    fig.add_trace(

    go.Scatter(x=days_extended_date, y=fit_active,

               marker = go.scatter.Marker(color= 'rgb(0, 0, 255)'),

               line={'dash':'solid', 'width':4},

               name = "Active - fit",

               showlegend=showlegend),

    row=row, col=col)

    

    fig.add_trace(

    go.Scatter(x=dates, y=recovered,

               marker = go.scatter.Marker(color= 'rgb(255, 255, 0)'),

               name = "Recovered",

               line={'dash':'solid', 'width':4},

               showlegend=showlegend),        

    row=row, col=col)

        

    fig.add_trace(

    go.Scatter(x=dates, y=deaths,

               marker = go.scatter.Marker(color= 'rgb(0, 0, 0)'),

               name = "Deaths",

               line={'dash':'solid', 'width':4},

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

    



def draw_figures(grid):

    fig = go.FigureWidget(make_subplots(

        rows=2, cols=2,

        shared_xaxes=False,

        horizontal_spacing = 0.05,

        vertical_spacing   = 0.05,

        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],

               [{'type': 'scatter'}, {'type': 'scatter'}]]))

    

    # adding surfaces to subplots.

    annotations = []

    countryFig(grid[0, 0].value, grid[1, 0].value, fig, 1, 1, annotations, showlegend=True, linestyle='dot')  #0,1

    countryFig(grid[0, 1].value, grid[1, 1].value, fig, 1, 2, annotations, linestyle='dot')                     #1,1   

    countryFig(grid[0, 2].value, grid[1, 2].value, fig, 2, 1, annotations, linestyle='dot')                  #0,0

    countryFig(grid[0, 3].value, grid[1, 3].value, fig, 2, 2, annotations, linestyle='dot')  #1,0

    

    fig.update_layout(

        title_text=r'COVID-19: Confirmed Total Cases vs Active Cases',

        autosize=False,

        height=900,

        width=900,

        #margin=dict(l=65, r=50, b=65, t=90),

        annotations = annotations

        )

    fig.update_xaxes(range=['2020-01-22','2020-07-31'])

    #fig.update_xaxes(rangeslider_visible=True)

    fig.show()
def provinces(country):

    province_list  = list(set(data.loc[data['Country/Region']==country]['Province/State']))

    return sorted([str(pi) for pi in province_list])



def Dropdowns(list_items, first, description='', disabled=False):

    return Dropdown(

        options=list_items,

        value=first,

        description=description,

        disabled=disabled,

        layout=Layout(width="50%"))



def province_observe(country, i, j):

    grid[i, j] = Dropdowns(provinces(country.new), provinces(country.new)[0])

    

def btn_eventhandler(obj):

    output.clear_output()

    with output:

        draw_figures(grid)
url = 'https://datahub.io/core/covid-19/r/time-series-19-covid-combined.csv'



headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:66.0) Gecko/20100101 Firefox/66.0"}

req = requests.get(url, headers=headers)

data_text = StringIO(req.text)



data = pd.read_csv(data_text)



countries = sorted(list(set(data['Country/Region'])))
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
grid = GridspecLayout(3, 4)



countries0 = ['Turkey', 'Iran', 'Germany', 'China']

province0 = ['nan', 'nan', 'nan', 'Hubei']



for j, cj in enumerate(countries0):

    grid[0, j] = Dropdowns(countries, cj)

    provinces_list =  provinces(grid[0, j].value)

    grid[1, j] = Dropdowns(provinces_list, province0[j])



grid[0, 0].observe(lambda country: province_observe(country, 1, 0), names='value')

grid[0, 1].observe(lambda country: province_observe(country, 1, 1), names='value')

grid[0, 2].observe(lambda country: province_observe(country, 1, 2), names='value')

grid[0, 3].observe(lambda country: province_observe(country, 1, 3), names='value')



grid[2, 0] = Button(description='Redraw')

grid[2, 0].on_click(btn_eventhandler)
display(grid)



output = Output()

display(output)



with output:

    draw_figures(grid)