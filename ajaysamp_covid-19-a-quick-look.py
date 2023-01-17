# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Standard plotly imports

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode



import plotly.express as px

import plotly.io as pio



# Using plotly + cufflinks in offline mode

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)
# read clean datatset that is updated every 24 hours

clean_data = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv',parse_dates=['Date'])
# rename column

clean_data = clean_data.rename(columns={'Country/Region':'Country'})
# overall cases growth 

overall_cases_death = clean_data.groupby('Date')['Date','Confirmed','Deaths','Recovered'].sum().reset_index()

india_cases_death = clean_data[clean_data.Country == 'India'].groupby('Date')['Date','Confirmed','Deaths','Recovered'].sum().reset_index()

rest_of_world_cases_death  = clean_data[clean_data.Country != 'India'].groupby('Date')['Date','Confirmed','Deaths','Recovered'].sum().reset_index()

italy_cases_death = clean_data[clean_data.Country == 'Italy'].groupby('Date')['Date','Confirmed','Deaths','Recovered'].sum().reset_index()
# plot overall confirmed cases

trace1 = go.Scatter(

                    x = overall_cases_death.Date,

                    y = overall_cases_death.Confirmed,

                    mode = "lines",

                    name = "",

                    text= 'Confirmed Cases',

                    line=dict(color='#4cff00', width=2),

                    fill='tozeroy')



data = [trace1]

layout = dict(title = {'text':'Growth of Confirmed Cases - Overall'},font=dict(color='white',family='Arial'),

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),

              yaxis =  {'showgrid': False},paper_bgcolor='#273746',

              plot_bgcolor='#273746')



fig = dict(data = data, layout = layout)



iplot(fig)
# India 

trace1 = go.Scatter(

                    x = india_cases_death.Date,

                    y = india_cases_death.Confirmed,

                    mode = "lines",

                    name = "India",

                    text = 'Confirmed Cases',

                    line=dict(color='#00ccff', width=2),

                    fill='tozeroy')



data = [trace1]

layout = dict(title = 'Growth of Confirmed Cases - India',font=dict(color='white',family='Arial'),

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),

              yaxis =  {'showgrid': False},paper_bgcolor='#273746',

              plot_bgcolor='#273746')

fig = dict(data = data, layout = layout)

iplot(fig)
# Italy

trace1 = go.Scatter(

                    x = italy_cases_death.Date,

                    y = italy_cases_death.Confirmed,

                    mode = "lines",

                    name = "Italy",

                    text = 'Confirmed Cases',

                    line=dict(color='#ff3300', width=2),fill='tozeroy')



data = [trace1]

layout = dict(title = 'Growth of Confirmed Cases - Italy',font=dict(color='white',family='Arial'),

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),

              yaxis =  {'showgrid': False},paper_bgcolor='#273746',

              plot_bgcolor='#273746')

        

fig = dict(data = data, layout = layout)

iplot(fig)
# top countries 10 countries with maximum confirmed cases 

last_date =  clean_data[clean_data.Date == max(clean_data.Date)]

top_10_countries_confirmed_cases = last_date.groupby('Country')['Confirmed','Deaths'].sum().reset_index().sort_values('Confirmed',ascending=False)[:10]
# Creating two subplots

from plotly.subplots import make_subplots



fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,

                    shared_yaxes=False, vertical_spacing=0.001)



fig.append_trace(go.Bar(

    x=top_10_countries_confirmed_cases[::-1].Confirmed,

    y=top_10_countries_confirmed_cases[::-1].Country,

    marker=dict(

        color='rgba(50, 171, 96, 0.6)',

        line=dict(

            color='rgba(50, 171, 96, 1.0)',

            width=1),

    ),

    name='Total Number of Confirmed Cases',

    orientation='h',

), 1, 1)



fig.append_trace(go.Scatter(

    x=top_10_countries_confirmed_cases[::-1].Deaths, y=top_10_countries_confirmed_cases[::-1].Country,

    mode='lines+markers',

    line_color='rgb(128, 0, 128)',

    name='Total Number of Deaths',

), 1, 2)



fig.update_layout(

    title='Covid-19 : Confirmed Cases & Total Deaths',

    yaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

        domain=[0, 0.85],

    ),

    yaxis2=dict(

        showgrid=False,

        showline=True,

        showticklabels=False,

        linecolor='rgba(102, 102, 102, 0.8)',

        linewidth=2,

        domain=[0, 0.85],

    ),

    xaxis=dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0, 0.42],

    ),

    xaxis2=dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0.47, 1],

        side='top',

        dtick=1000,

    ),

    legend=dict(x=0.029, y=1.038, font_size=10),

    margin=dict(l=100, r=20, t=70, b=70),

    paper_bgcolor='rgb(248, 248, 255)',

    plot_bgcolor='rgb(248, 248, 255)',

)



annotations = []



y_s = np.round(top_10_countries_confirmed_cases.Confirmed, decimals=0)

y_nw = np.rint(top_10_countries_confirmed_cases.Deaths)



# Adding labels

for ydn, yd, xd in zip(y_nw, y_s, top_10_countries_confirmed_cases.Country):

    # labeling the scatter deaths

    annotations.append(dict(xref='x2', yref='y2',

                            y=xd, x=ydn + 250,

                            text='{:,.0f}'.format(ydn),

                            font=dict(family='Arial', size=12,

                                      color='rgb(128, 0, 128)'),

                            showarrow=False))

    # labeling the bar confirmed cases

    annotations.append(dict(xref='x1', yref='y1',

                            y=xd, x=yd + 5000,

                            text='{:,.0f}'.format(yd),

                            font=dict(family='Arial', size=12,

                                      color='rgb(50, 171, 96)'),

                            showarrow=False))



fig.update_layout(annotations=annotations)



fig.show()
# plot overall confirmed fatalities

trace1 = go.Scatter(

                    x = overall_cases_death.Date,

                    y = overall_cases_death.Deaths,

                    mode = "lines",

                    name = "",

                    text= 'Confirmed Fatalities', line=dict(color='#4cff00', width=2),fill='tozeroy')



data = [trace1]

layout = dict(title = 'Growth of Confirmed Fatalities - Overall',font=dict(color='white',family='Arial'),

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),

              yaxis =  {'showgrid': False},paper_bgcolor='#273746',

              plot_bgcolor='#273746')



fig = dict(data = data, layout = layout)

iplot(fig)
# plot India fatalities

trace1 = go.Scatter(

                    x = india_cases_death.Date,

                    y = india_cases_death.Deaths,

                    mode = "lines",

                    name = "",

                    text= 'Confirmed Fatalities',line=dict(color='#00ccff', width=2),fill='tozeroy')



data = [trace1]

layout = dict(title = 'Growth of Confirmed Fatalities - India',font=dict(color='white',family='Arial'),

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),

              yaxis =  {'showgrid': False},paper_bgcolor='#273746',

              plot_bgcolor='#273746')



fig = dict(data = data, layout = layout)

iplot(fig)
# plot Italy fatalities

trace1 = go.Scatter(

                    x = italy_cases_death.Date,

                    y = italy_cases_death.Deaths,

                    mode = "lines",

                    name = "",

                    text= 'Confirmed Fatalities', line=dict(color='#ff3300', width=2),fill='tozeroy')



data = [trace1]

layout = dict(title = 'Growth of Confirmed Fatalities - Italy',font=dict(color='white',family='Arial'),

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),

              yaxis =  {'showgrid': False},paper_bgcolor='#273746',

              plot_bgcolor='#273746')

fig = dict(data = data, layout = layout)

iplot(fig)
latest_deaths = clean_data[clean_data.Date == max(clean_data.Date)]

top_10_countries_confirmed_deaths = latest_deaths.groupby('Country')['Deaths'].sum().reset_index().sort_values('Deaths', ascending=False)[:10]
fig = px.bar(top_10_countries_confirmed_deaths[::-1], x= 'Deaths', y='Country', orientation='h',text='Deaths',

             title='Confirmed Fatalities - Top 10 Countries',template="plotly_dark")

fig.show()
# plot overall recovered cases

trace1 = go.Scatter(

                    x = overall_cases_death.Date,

                    y = overall_cases_death.Recovered,

                    mode = "lines",

                    name = "",

                    text= 'Recoveries',line=dict(color='#4cff00', width=2),fill='tozeroy')



data = [trace1]

layout = dict(title = 'Recovered Cases - Overall',font=dict(color='white',family='Arial'),

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),

              yaxis =  {'showgrid': False},paper_bgcolor='#273746',

              plot_bgcolor='#273746')



fig = dict(data = data, layout = layout)

iplot(fig)
# plot recovered cases in India

trace1 = go.Scatter(

                    x = india_cases_death.Date,

                    y = india_cases_death.Recovered,

                    mode = "lines",

                    name = "",

                    text= 'Recoveries',line=dict(color='#00ccff', width=2),fill='tozeroy')



data = [trace1]

layout = dict(title = 'Recovered Cases - India',font=dict(color='white',family='Arial'),

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),

              yaxis =  {'showgrid': False},paper_bgcolor='#273746',

              plot_bgcolor='#273746')



fig = dict(data = data, layout = layout)

iplot(fig)
# plot recovered cases in Italy

trace1 = go.Scatter(

                    x = italy_cases_death.Date,

                    y = italy_cases_death.Recovered,

                    mode = "lines",

                    name = "",

                    text= 'Recoveries',line=dict(color='#ff3300', width=2),fill='tozeroy')



data = [trace1]

layout = dict(title = 'Recovered Cases - Italy',font=dict(color='white',family='Arial'),

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),

              yaxis =  {'showgrid': False},paper_bgcolor='#273746',

              plot_bgcolor='#273746')



fig = dict(data = data, layout = layout)

iplot(fig)
last_recovery = clean_data[clean_data.Date == max(clean_data.Date)]

top_10_countries_recovery = last_recovery.groupby('Country')['Recovered'].sum().reset_index().sort_values('Recovered', ascending=False)[:10]
fig = px.bar(top_10_countries_recovery[::-1], x= 'Recovered', y='Country', orientation='h',text='Recovered',

             title='Confirmed Recoveries - Top 10 Countries',template="plotly_dark")

fig.show()
# add active cases columns

clean_data['Active'] = clean_data['Confirmed'] - (clean_data['Recovered'] + clean_data['Deaths'])
# active cases growth

active_case_growth = clean_data.groupby('Date')['Active'].sum().reset_index()
# plot active case growth

trace1 = go.Scatter(

                    x = active_case_growth.Date,

                    y = active_case_growth.Active,

                    mode = "lines",

                    name = "",

                    text= 'Active Cases',line=dict(color='#4cff00', width=2),fill='tozeroy')



data = [trace1]



layout = dict(title = 'Active Case Growth - Overall',font=dict(color='white',family='Arial'),

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),

              yaxis =  {'showgrid': False},paper_bgcolor='#273746',

              plot_bgcolor='#273746')



fig = dict(data = data, layout = layout)

iplot(fig)
india_active_cases = clean_data[clean_data.Country == 'India'].groupby('Date')['Active'].sum().reset_index()

italy_active_cases = clean_data[clean_data.Country == 'Italy'].groupby('Date')['Active'].sum().reset_index()

china_active_cases = clean_data[clean_data.Country == 'China'].groupby('Date')['Active'].sum().reset_index()
# plot active case growth - India

trace1 = go.Scatter(

                    x = india_active_cases.Date,

                    y = india_active_cases.Active,

                    mode = "lines",

                    name = "",

                    text= 'Active Cases', line=dict(color='#00ccff', width=2),fill='tozeroy')



data = [trace1]

layout = dict(title = 'Active Case Growth - India',font=dict(color='white',family='Arial'),

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),

              yaxis =  {'showgrid': False},paper_bgcolor='#273746',

              plot_bgcolor='#273746')



fig = dict(data = data, layout = layout)

iplot(fig)
# plot active case growth - Italy

trace1 = go.Scatter(

                    x = italy_active_cases.Date,

                    y = italy_active_cases.Active,

                    mode = "lines",

                    name = "",

                    text= 'Active Cases', line=dict(color='#ff3300', width=2),fill='tozeroy')



data = [trace1]

layout = dict(title = 'Active Case Growth - Italy',font=dict(color='white',family='Arial'),

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),

              yaxis =  {'showgrid': False},paper_bgcolor='#273746',

              plot_bgcolor='#273746')





fig = dict(data = data, layout = layout)

iplot(fig)
# plot active case growth - India

trace1 = go.Scatter(

                    x = china_active_cases.Date,

                    y = china_active_cases.Active,

                    mode = "lines",

                    name = "",

                    text= 'Active Cases', line=dict(color='#F1C40F', width=2),fill='tozeroy')



data = [trace1]

layout = dict(title = 'Active Case Growth - China',font=dict(color='white',family='Arial'),

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),

              yaxis =  {'showgrid': False},paper_bgcolor='#273746',

              plot_bgcolor='#273746')

fig = dict(data = data, layout = layout)

iplot(fig)
last_rec_death = clean_data[clean_data.Date == max(clean_data.Date)]

recovery_death_rates = last_rec_death.groupby('Country')['Confirmed','Deaths','Recovered'].sum().reset_index()
# add death rate and recovery rate

recovery_death_rates['Death_Rate'] = (recovery_death_rates['Deaths']/recovery_death_rates['Confirmed']) * 100

recovery_death_rates['Recovery_Rate'] = (recovery_death_rates['Recovered']/recovery_death_rates['Confirmed']) * 100
top_10_death_rates = recovery_death_rates.sort_values('Confirmed', ascending=False)[:10]

top_10_recovery_rates = recovery_death_rates.sort_values('Confirmed', ascending=False)[:10]
# countries with recovery rates



fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,

                    shared_yaxes=False, vertical_spacing=0.001)



fig.append_trace(go.Bar(

    x=top_10_recovery_rates[::-1].Confirmed,

    y=top_10_recovery_rates[::-1].Country,

    marker=dict(

        color='rgba(50, 171, 96, 0.6)',

        line=dict(

            color='rgba(50, 171, 96, 1.0)',

            width=1),

    ),

    name='Total Number of Confirmed Cases',

    orientation='h',

), 1, 1)



fig.append_trace(go.Scatter(

    x=top_10_recovery_rates[::-1].Recovery_Rate, 

    y=top_10_recovery_rates[::-1].Country,

    mode='lines+markers',

    line_color='rgb(128, 0, 128)',

    name='Recovery Rate',

), 1, 2)



fig.update_layout(

    title='Covid-19 : Confirmed Cases & Recovery Rates',

    yaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

        domain=[0, 0.85],

    ),

    yaxis2=dict(

        showgrid=False,

        showline=True,

        showticklabels=False,

        linecolor='rgba(102, 102, 102, 0.8)',

        linewidth=2,

        domain=[0, 0.85],

    ),

    xaxis=dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0, 0.42],

    ),

    xaxis2=dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0.47, 1],

        side='top',

        dtick=20,

    ),

    legend=dict(x=0.029, y=1.038, font_size=10),

    margin=dict(l=100, r=20, t=70, b=70),

    paper_bgcolor='rgb(248, 248, 255)',

    plot_bgcolor='rgb(248, 248, 255)',

)



annotations = []



y_s = np.round(top_10_recovery_rates.Confirmed, decimals=0)

y_nw = np.rint(top_10_recovery_rates.Recovery_Rate)



# Adding labels

for ydn, yd, xd in zip(y_nw, y_s, top_10_countries_confirmed_cases.Country):

    # labeling the scatter deaths

    annotations.append(dict(xref='x2', yref='y2',

                            y=xd, x=ydn + 10,

                            text= str(ydn) + '%',

                            font=dict(family='Arial', size=12,

                                      color='rgb(128, 0, 128)'),

                            showarrow=False))

    # labeling the bar confirmed cases

    annotations.append(dict(xref='x1', yref='y1',

                            y=xd, x=yd + 5000,

                            text='{:,.0f}'.format(yd),

                            font=dict(family='Arial', size=12,

                                      color='rgb(50, 171, 96)'),

                            showarrow=False))



fig.update_layout(annotations=annotations)



fig.show()
# confirmed cases & mortality rates



fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,

                    shared_yaxes=False, vertical_spacing=0.001)



fig.append_trace(go.Bar(

    x=top_10_death_rates[::-1].Confirmed,

    y=top_10_death_rates[::-1].Country,

    marker=dict(

        color='rgba(50, 171, 96, 0.6)',

        line=dict(

            color='rgba(50, 171, 96, 1.0)',

            width=1),

    ),

    name='Total Number of Confirmed Cases',

    orientation='h',

), 1, 1)



fig.append_trace(go.Scatter(

    x=top_10_death_rates[::-1].Death_Rate, 

    y=top_10_death_rates[::-1].Country,

    mode='lines+markers',

    line_color='rgb(128, 0, 128)',

    name='Recovery Rate',

), 1, 2)



fig.update_layout(

    title='Covid-19 : Confirmed Cases & Mortality Rates',

    yaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

        domain=[0, 0.85],

    ),

    yaxis2=dict(

        showgrid=False,

        showline=True,

        showticklabels=False,

        linecolor='rgba(102, 102, 102, 0.8)',

        linewidth=2,

        domain=[0, 0.85],

    ),

    xaxis=dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0, 0.42],

    ),

    xaxis2=dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0.47, 1],

        side='top',

        dtick=2,

    ),

    legend=dict(x=0.029, y=1.038, font_size=10),

    margin=dict(l=100, r=20, t=70, b=70),

    paper_bgcolor='rgb(248, 248, 255)',

    plot_bgcolor='rgb(248, 248, 255)',

)



annotations = []



y_s = np.round(top_10_death_rates.Confirmed, decimals=0)

y_nw = np.rint(top_10_death_rates.Death_Rate)



# Adding labels

for ydn, yd, xd in zip(y_nw, y_s, top_10_death_rates.Country):

    # labeling the scatter deaths

    annotations.append(dict(xref='x2', yref='y2',

                            y=xd, x=ydn + 0.8,

                            text= str(ydn) + '%',

                            font=dict(family='Arial', size=12,

                                      color='rgb(128, 0, 128)'),

                            showarrow=False))

    # labeling the bar confirmed cases

    annotations.append(dict(xref='x1', yref='y1',

                            y=xd, x=yd + 5000,

                            text='{:,.0f}'.format(yd),

                            font=dict(family='Arial', size=12,

                                      color='rgb(50, 171, 96)'),

                            showarrow=False))



fig.update_layout(annotations=annotations)



fig.show()
# best recovery rates - more than 100 cases 

best_rec_rate = recovery_death_rates[recovery_death_rates.Confirmed >= 100].sort_values('Recovery_Rate', ascending=False)[:10]

best_rec_rate['Recovery_Rate'] = np.round(best_rec_rate['Recovery_Rate'],1)



# Worst Mortality Rates

worst_mort_rate = recovery_death_rates[recovery_death_rates.Confirmed >= 100].sort_values('Death_Rate', ascending=False)[:10]

worst_mort_rate['Death_Rate'] = np.round(worst_mort_rate['Death_Rate'],1)
fig = px.bar(best_rec_rate[::-1], x= 'Recovery_Rate', y='Country', 

             orientation='h',text='Recovery_Rate',title='Countries with best recovery rates (> 100 Confirmed Cases)',

             labels={'Recovery_Rate':'Recovery Rate %'}, template="plotly_dark")

fig.show()
fig = px.bar(worst_mort_rate[::-1], x= 'Death_Rate', y='Country', orientation='h',text='Death_Rate',

             title='Countries with worst mortality rates (> 100 Confirmed Cases)', 

             labels={'Death_Rate':'Mortality Rate %'}, template="plotly_dark")

fig.show()