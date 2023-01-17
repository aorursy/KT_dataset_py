import pandas as pd

import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

pd.set_option('display.max_columns',1000)

import warnings

warnings. simplefilter(action='ignore', category=Warning) 
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
gen = pd.read_csv('/kaggle/input/daily-power-generation-in-india-20172020/file.csv')

gen.head()
states = pd.read_csv('/kaggle/input/daily-power-generation-in-india-20172020/State_Region_corrected.csv')

states.head()
share = states.groupby('Region').agg('sum')

share = share.reset_index()

share
fig=px.bar(states,

           x='State / Union territory (UT)',

           y='Area (km2)',

           hover_data=['Area (km2)'],

           color='State / Union territory (UT)',

           )

fig.update_layout(title_text='Statewise Area(km2)')

fig.show()
fig=px.bar(states,

           x='State / Union territory (UT)',

           y='National Share (%)',

           hover_data=['National Share (%)'],

           color='State / Union territory (UT)',

           )

fig.update_layout(title_text='Statewise Generation Share (%)')

fig.show()
fig=make_subplots(rows=2, cols=2, 

                  subplot_titles=('Regionwise Area Share', 'Regionwise Area Share',

                                  'Regionwise Production Share', 'Regionwise Production Share'),

                 specs=[[{"type": "bar"}, {"type": "pie"}],

                       [{"type": "bar"}, {"type": "pie"}]])



fig.add_trace(

    go.Bar(x=share['Region'],

           y=share['Area (km2)'],

           name='Area'),

    row=1, col=1

)

fig.add_trace(

    go.Pie(labels=share['Region'],

           values=share['Area (km2)']),

    row=1, col=2

)

fig.add_trace(

    go.Bar(x=share['Region'],

           y=share['National Share (%)'],

           name='Production Share'),

    row=2, col=1

)

fig.add_trace(

    go.Pie(labels=share['Region'],

           values=share['Area (km2)']),

    row=2, col=2

)



fig.show()
gen.dtypes
gen.isnull().sum()
for col in gen.columns:

    if 'Nuclear' in col:

        gen[col] = gen[col].fillna(0)

gen.head()
gen['Thermal Generation Actual (in MU)'] = gen['Thermal Generation Actual (in MU)'].str.replace(',','')

gen['Thermal Generation Estimated (in MU)'] = gen['Thermal Generation Estimated (in MU)'].str.replace(',','')
gen['Thermal Generation Actual (in MU)'] = gen['Thermal Generation Actual (in MU)'].apply(pd.to_numeric, errors='coerce')

gen['Thermal Generation Estimated (in MU)'] = gen['Thermal Generation Estimated (in MU)'].apply(pd.to_numeric, errors = 'coerce')

gen['Date'] = pd.to_datetime(gen['Date'], errors='ignore')

gen.head()
actual_cols = []

for col in gen.columns:

    if 'Actual' in col:

        actual_cols.append(col)

gen['Total Generation Actual (in MU)'] = 0

for col in actual_cols:

    gen['Total Generation Actual (in MU)'] = gen['Total Generation Actual (in MU)'] + gen[col]

gen.head()
estimated_col = []

for col in gen.columns:

    if 'Estimated' in col:

        estimated_col.append(col)

gen['Total Generation Estimated (in MU)'] = 0

for col in estimated_col:

    gen['Total Generation Estimated (in MU)'] = gen['Total Generation Estimated (in MU)'] + gen[col]

gen.head()
gen['Net Surplus'] = gen['Total Generation Actual (in MU)'] - gen['Total Generation Estimated (in MU)']

gen.head()
gen['Year'] = pd.DatetimeIndex(gen['Date']).year

gen['Month'] = pd.DatetimeIndex(gen['Date']).month
gen['Surplus in Thermal Generation'] = gen['Thermal Generation Actual (in MU)'] - gen['Thermal Generation Estimated (in MU)']

gen['Surplus in Nuclear Generation'] = gen['Nuclear Generation Actual (in MU)'] - gen['Nuclear Generation Estimated (in MU)']

gen['Surplus in Hydro Generation'] = gen['Hydro Generation Actual (in MU)'] - gen['Hydro Generation Estimated (in MU)']
gen['Thermal Generation (in %)'] = (gen['Thermal Generation Actual (in MU)']/gen['Total Generation Actual (in MU)'])*100

gen['Nuclear Generation (in %)'] = (gen['Nuclear Generation Actual (in MU)']/gen['Total Generation Actual (in MU)'])*100

gen['Hydro Generation (in %)'] = (gen['Hydro Generation Actual (in MU)']/gen['Total Generation Actual (in MU)'])*100
gen = gen[['Date', 'Month', 'Year', 'Region', 'Thermal Generation Actual (in MU)', 'Thermal Generation Estimated (in MU)', 

           'Thermal Generation (in %)', 'Surplus in Thermal Generation', 'Nuclear Generation Actual (in MU)', 

           'Nuclear Generation Estimated (in MU)', 'Nuclear Generation (in %)', 'Surplus in Nuclear Generation', 

           'Hydro Generation Actual (in MU)', 'Hydro Generation Estimated (in MU)', 'Hydro Generation (in %)',

           'Surplus in Hydro Generation', 'Total Generation Actual (in MU)', 'Net Surplus']]
gen.head()
Northern = gen[gen['Region'] == 'Northern']

Western = gen[gen['Region'] == 'Western']

Southern = gen[gen['Region'] == 'Southern']

Eastern = gen[gen['Region'] == 'Eastern']

NE = gen[gen['Region'] == 'NorthEastern']
fig=go.Figure()



fig.add_trace(

    go.Scatter(x=Northern['Date'],

               y=Northern['Thermal Generation Actual (in MU)'],

               marker_color='rgb(0,0,204)',

               name='Northern Region')

)



fig.add_trace(

    go.Scatter(x=Western['Date'],

               y=Western['Thermal Generation Actual (in MU)'],

               marker_color='rgb(204,0,0)',

               name='Western Region')

)



fig.add_trace(

    go.Scatter(x=Southern['Date'],

               y=Southern['Thermal Generation Actual (in MU)'],

               marker_color='rgb(102,102,51)',

               name='Southern Region')

)



fig.add_trace(

    go.Scatter(x=Eastern['Date'],

               y=Eastern['Thermal Generation Actual (in MU)'],

               marker_color='rgb(102,153,153)',

               name='Eastern Region')

)



fig.add_trace(

    go.Scatter(x=NE['Date'],

               y=NE['Thermal Generation Actual (in MU)'],

               marker_color='rgb(0,204,153)',

               name='North-East Region')

)





fig.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )



fig.update_layout(title='Thermal Generation Actual (in MU)')

fig.show()
fig=go.Figure()



fig.add_trace(

    go.Scatter(x=Northern['Date'],

               y=Northern['Nuclear Generation Actual (in MU)'],

               marker_color='rgb(0,0,204)',

               name='Northern Region')

)



fig.add_trace(

    go.Scatter(x=Western['Date'],

               y=Western['Nuclear Generation Actual (in MU)'],

               marker_color='rgb(204,0,0)',

               name='Western Region')

)



fig.add_trace(

    go.Scatter(x=Southern['Date'],

               y=Southern['Nuclear Generation Actual (in MU)'],

               marker_color='rgb(102,102,51)',

               name='Southern Region')

)



fig.add_trace(

    go.Scatter(x=Eastern['Date'],

               y=Eastern['Nuclear Generation Actual (in MU)'],

               marker_color='rgb(102,153,153)',

               name='Eastern Region')

)



fig.add_trace(

    go.Scatter(x=NE['Date'],

               y=NE['Nuclear Generation Actual (in MU)'],

               marker_color='rgb(0,204,153)',

               name='North-East Region')

)





fig.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )



fig.update_layout(title='Nuclear Generation Actual (in MU)')

fig.show()
fig=go.Figure()



fig.add_trace(

    go.Scatter(x=Northern['Date'],

               y=Northern['Hydro Generation Actual (in MU)'],

               marker_color='rgb(0, 0, 204)',

               name='Northern Region')

)



fig.add_trace(

    go.Scatter(x=Western['Date'],

               y=Western['Hydro Generation Actual (in MU)'],

               marker_color='rgb(204,0,0)',

               name='Western Region')

)



fig.add_trace(

    go.Scatter(x=Southern['Date'],

               y=Southern['Hydro Generation Actual (in MU)'],

               marker_color='rgb(102,102,51)',

               name='Southern Region')

)



fig.add_trace(

    go.Scatter(x=Eastern['Date'],

               y=Eastern['Hydro Generation Actual (in MU)'],

               marker_color='rgb(102,153,153)',

               name='Eastern Region')

)



fig.add_trace(

    go.Scatter(x=NE['Date'],

               y=NE['Hydro Generation Actual (in MU)'],

               marker_color='rgb(0,204,153)',

               name='North-East Region')

)





fig.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )



fig.update_layout(title='Hydro Generation Actual (in MU)')

fig.show()
fig=go.Figure()



fig.add_trace(

    go.Scatter(x=Northern['Date'],

               y=Northern['Total Generation Actual (in MU)'],

               marker_color='rgb(0, 0, 204)',

               name='Northern Region')

)



fig.add_trace(

    go.Scatter(x=Western['Date'],

               y=Western['Total Generation Actual (in MU)'],

               marker_color='rgb(204,0,0)',

               name='Western Region')

)



fig.add_trace(

    go.Scatter(x=Southern['Date'],

               y=Southern['Total Generation Actual (in MU)'],

               marker_color='rgb(102,102,51)',

               name='Southern Region')

)



fig.add_trace(

    go.Scatter(x=Eastern['Date'],

               y=Eastern['Total Generation Actual (in MU)'],

               marker_color='rgb(102,153,153)',

               name='Eastern Region')

)



fig.add_trace(

    go.Scatter(x=NE['Date'],

               y=NE['Total Generation Actual (in MU)'],

               marker_color='rgb(0,204,153)',

               name='North-East Region')

)





fig.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )



fig.update_layout(title='Total Actual Generation (in MU)')

fig.show()
monthly_generation = gen[['Month', 'Year', 'Total Generation Actual (in MU)']].groupby(['Year', 'Month']).agg(sum)

monthly_generation.style.background_gradient(cmap='Blues_r')
monthly_generation = monthly_generation.reset_index(['Year','Month'])

monthly_generation['Year'] = monthly_generation['Year'].astype(str)

monthly_generation['Month'] = monthly_generation['Month'].astype(str)
monthly_generation['Month/Year'] = monthly_generation['Month'] + '/' + monthly_generation['Year']

fig = go.Figure()

fig.add_trace(

    go.Scatter(x=monthly_generation['Month/Year'],

               y=monthly_generation['Total Generation Actual (in MU)'],

               mode='lines+markers',

               marker_color='rgb(0, 0, 204)',

               name='Total Generation Actual (in MU) line')

)

fig.add_trace(

    go.Bar(x=monthly_generation['Month/Year'],

           y=monthly_generation['Total Generation Actual (in MU)'],

           marker_color='rgb(204, 0, 0)',

           name='Total Generation Actual (in MU) bar')

)



fig.show()
Northern['Thermal Generation Actual daily difference (in MU)'] = Northern['Thermal Generation Actual (in MU)'].diff()

Southern['Thermal Generation Actual daily difference (in MU)'] = Southern['Thermal Generation Actual (in MU)'].diff()

Western['Thermal Generation Actual daily difference (in MU)'] = Western['Thermal Generation Actual (in MU)'].diff()

Eastern['Thermal Generation Actual daily difference (in MU)'] = Eastern['Thermal Generation Actual (in MU)'].diff()

NE['Thermal Generation Actual daily difference (in MU)'] = NE['Thermal Generation Actual (in MU)'].diff()
Northern['Nuclear Generation Actual daily difference (in MU)'] = Northern['Nuclear Generation Actual (in MU)'].diff()

Southern['Nuclear Generation Actual daily difference (in MU)'] = Southern['Nuclear Generation Actual (in MU)'].diff()

Western['Nuclear Generation Actual daily difference (in MU)'] = Western['Nuclear Generation Actual (in MU)'].diff()

Eastern['Nuclear Generation Actual daily difference (in MU)'] = Eastern['Nuclear Generation Actual (in MU)'].diff()

NE['Nuclear Generation Actual daily difference (in MU)'] = NE['Nuclear Generation Actual (in MU)'].diff()
Northern['Hydro Generation Actual daily difference (in MU)'] = Northern['Hydro Generation Actual (in MU)'].diff()

Southern['Hydro Generation Actual daily difference (in MU)'] = Southern['Hydro Generation Actual (in MU)'].diff()

Western['Hydro Generation Actual daily difference (in MU)'] = Western['Hydro Generation Actual (in MU)'].diff()

Eastern['Hydro Generation Actual daily difference (in MU)'] = Eastern['Hydro Generation Actual (in MU)'].diff()

NE['Hydro Generation Actual daily difference (in MU)'] = NE['Hydro Generation Actual (in MU)'].diff()
Northern['Net Surplus'] = Northern['Net Surplus'].diff()

Southern['Net Surplus'] = Southern['Net Surplus'].diff()

Western['Net Surplus'] = Western['Net Surplus'].diff()

Eastern['Net Surplus'] = Eastern['Net Surplus'].diff()

NE['Net Surplus'] = NE['Net Surplus'].diff()
Northern.fillna(0, inplace = True)

Southern.fillna(0, inplace = True)

Western.fillna(0, inplace = True)

Eastern.fillna(0, inplace=True)

NE.fillna(0, inplace = True)
fig_one = go.Figure(go.Scatter(x=Northern['Date'],

               y=Northern['Thermal Generation Actual daily difference (in MU)'],

               marker_color='rgb(0, 0, 204)',

               name='Northern Region')

)

fig_one.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_one.update_layout(yaxis=dict(range=[-150,150]), title_text='Northern Region')

fig_one.show()



fig_two = go.Figure(go.Scatter(x=Southern['Date'],

               y=Southern['Thermal Generation Actual daily difference (in MU)'],

               marker_color='rgb(0, 0, 204)',

               name='Southern Region')

)

fig_two.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_two.update_layout(yaxis=dict(range=[-150,150]), title_text='Southern Region')

fig_two.show()



fig_three = go.Figure(go.Scatter(x=Western['Date'],

               y=Western['Thermal Generation Actual daily difference (in MU)'],

               marker_color='rgb(0, 0, 204)',

               name='Western Region')

)

fig_three.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_three.update_layout(yaxis=dict(range=[-150,150]), title_text='Western Region')

fig_three.show()



fig_four = go.Figure(go.Scatter(x=Eastern['Date'],

               y=Eastern['Thermal Generation Actual daily difference (in MU)'],

               marker_color='rgb(0, 0, 204)',

               name='Eastern Region')

)

fig_four.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_four.update_layout(yaxis=dict(range=[-150,150]), title_text='Eastern Region')

fig_four.show()



fig_five = go.Figure(go.Scatter(x=NE['Date'],

               y=NE['Thermal Generation Actual daily difference (in MU)'],

               marker_color='rgb(0, 0, 204)',

               name='NE Region')

)

fig_five.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_five.update_layout(yaxis=dict(range=[-200,200]), title_text='NE Region')

fig_five.show()
fig_one = go.Figure(go.Scatter(x=Northern['Date'],

               y=Northern['Nuclear Generation Actual daily difference (in MU)'],

               marker_color='rgb(0, 0, 204)',

               name='Northern Region')

)

fig_one.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_one.update_layout(yaxis=dict(range=[-150,150]), title_text='Northern Region')

fig_one.show()



fig_two = go.Figure(go.Scatter(x=Southern['Date'],

               y=Southern['Nuclear Generation Actual daily difference (in MU)'],

               marker_color='rgb(0, 0, 204)',

               name='Southern Region')

)

fig_two.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_two.update_layout(yaxis=dict(range=[-150,150]), title_text='Southern Region')

fig_two.show()



fig_three = go.Figure(go.Scatter(x=Western['Date'],

               y=Western['Nuclear Generation Actual daily difference (in MU)'],

               marker_color='rgb(0, 0, 204)',

               name='Western Region')

)

fig_three.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_three.update_layout(yaxis=dict(range=[-150,150]), title_text='Western Region')

fig_three.show()



fig_four = go.Figure(go.Scatter(x=Eastern['Date'],

               y=Eastern['Nuclear Generation Actual daily difference (in MU)'],

               marker_color='rgb(0, 0, 204)',

               name='Eastern Region')

)

fig_four.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_four.update_layout(yaxis=dict(range=[-150,150]), title_text='Eastern Region')

fig_four.show()



fig_five = go.Figure(go.Scatter(x=NE['Date'],

               y=NE['Nuclear Generation Actual daily difference (in MU)'],

               marker_color='rgb(0, 0, 204)',

               name='NE Region')

)

fig_five.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_five.update_layout(yaxis=dict(range=[-150,150]), title_text='NE Region')

fig_five.show()
fig_one = go.Figure(go.Scatter(x=Northern['Date'],

               y=Northern['Hydro Generation Actual daily difference (in MU)'],

               marker_color='rgb(0, 0, 204)',

               name='Northern Region')

)

fig_one.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_one.update_layout(yaxis=dict(range=[-150,150]), title_text='Northern Region')

fig_one.show()



fig_two = go.Figure(go.Scatter(x=Southern['Date'],

               y=Southern['Hydro Generation Actual daily difference (in MU)'],

               marker_color='rgb(0, 0, 204)',

               name='Southern Region')

)

fig_two.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_two.update_layout(yaxis=dict(range=[-150,150]), title_text='Southern Region')

fig_two.show()



fig_three = go.Figure(go.Scatter(x=Western['Date'],

               y=Western['Hydro Generation Actual daily difference (in MU)'],

               marker_color='rgb(0, 0, 204)',

               name='Western Region')

)

fig_three.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_three.update_layout(yaxis=dict(range=[-150,150]), title_text='Western Region')

fig_three.show()



fig_four = go.Figure(go.Scatter(x=Eastern['Date'],

               y=Eastern['Hydro Generation Actual daily difference (in MU)'],

               marker_color='rgb(0, 0, 204)',

               name='Eastern Region')

)

fig_four.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_four.update_layout(yaxis=dict(range=[-150,150]), title_text='Eastern Region')

fig_four.show()



fig_five = go.Figure(go.Scatter(x=NE['Date'],

               y=NE['Hydro Generation Actual daily difference (in MU)'],

               marker_color='rgb(0, 0, 204)',

               name='NE Region')

)

fig_five.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_five.update_layout(yaxis=dict(range=[-150,150]), title_text='NE Region')

fig_five.show()
fig_one = go.Figure(go.Scatter(x=Northern['Date'],

               y=Northern['Nuclear Generation (in %)'],

               marker_color='rgb(0, 0, 204)',

               name='Northern Region')

)

fig_one.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_one.update_layout(yaxis=dict(range=[-10,10]), title_text='Northern Region')

fig_one.show()



fig_two = go.Figure(go.Scatter(x=Southern['Date'],

               y=Southern['Nuclear Generation (in %)'],

               marker_color='rgb(0, 0, 204)',

               name='Southern Region')

)

fig_two.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_two.update_layout(yaxis=dict(range=[-15,15]), title_text='Southern Region')

fig_two.show()



fig_three = go.Figure(go.Scatter(x=Western['Date'],

               y=Western['Nuclear Generation (in %)'],

               marker_color='rgb(0, 0, 204)',

               name='Western Region')

)

fig_three.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_three.update_layout(yaxis=dict(range=[-7,7]), title_text='Western Region')

fig_three.show()



fig_four = go.Figure(go.Scatter(x=Eastern['Date'],

               y=Eastern['Nuclear Generation (in %)'],

               marker_color='rgb(0, 0, 204)',

               name='Eastern Region')

)

fig_four.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_four.update_layout(yaxis=dict(range=[-7,7]), title_text='Eastern Region')

fig_four.show()



fig_five = go.Figure(go.Scatter(x=NE['Date'],

               y=NE['Nuclear Generation (in %)'],

               marker_color='rgb(0, 0, 204)',

               name='NE Region')

)

fig_five.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_five.update_layout(yaxis=dict(range=[-7,7]), title_text='NE Region')

fig_five.show()
fig_one = go.Figure(go.Scatter(x=Northern['Date'],

               y=Northern['Thermal Generation (in %)'],

               marker_color='rgb(0, 0, 204)',

               name='Northern Region')

)

fig_one.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_one.update_layout(yaxis=dict(range=[-100,100]), title_text='Northern Region')

fig_one.show()



fig_two = go.Figure(go.Scatter(x=Southern['Date'],

               y=Southern['Thermal Generation (in %)'],

               marker_color='rgb(0, 0, 204)',

               name='Southern Region')

)

fig_two.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_two.update_layout(yaxis=dict(range=[-100,100]), title_text='Southern Region')

fig_two.show()



fig_three = go.Figure(go.Scatter(x=Western['Date'],

               y=Western['Thermal Generation (in %)'],

               marker_color='rgb(0, 0, 204)',

               name='Western Region')

)

fig_three.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_three.update_layout(yaxis=dict(range=[-100,100]), title_text='Western Region')

fig_three.show()



fig_four = go.Figure(go.Scatter(x=Eastern['Date'],

               y=Eastern['Thermal Generation (in %)'],

               marker_color='rgb(0, 0, 204)',

               name='Eastern Region')

)

fig_four.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_four.update_layout(yaxis=dict(range=[-100,100]), title_text='Eastern Region')

fig_four.show()



fig_five = go.Figure(go.Scatter(x=NE['Date'],

               y=NE['Thermal Generation (in %)'],

               marker_color='rgb(0, 0, 204)',

               name='NE Region')

)

fig_five.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_five.update_layout(yaxis=dict(range=[-100,100]), title_text='NE Region')

fig_five.show()
fig_one = go.Figure(go.Scatter(x=Northern['Date'],

               y=Northern['Hydro Generation (in %)'],

               marker_color='rgb(0, 0, 204)',

               name='Northern Region')

)

fig_one.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_one.update_layout(yaxis=dict(range=[-50,50]), title_text='Northern Region')

fig_one.show()



fig_two = go.Figure(go.Scatter(x=Southern['Date'],

               y=Southern['Hydro Generation (in %)'],

               marker_color='rgb(0, 0, 204)',

               name='Southern Region')

)

fig_two.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_two.update_layout(yaxis=dict(range=[-50,50]), title_text='Southern Region')

fig_two.show()



fig_three = go.Figure(go.Scatter(x=Western['Date'],

               y=Western['Hydro Generation (in %)'],

               marker_color='rgb(0, 0, 204)',

               name='Western Region')

)

fig_three.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_three.update_layout(yaxis=dict(range=[-50,50]), title_text='Western Region')

fig_three.show()



fig_four = go.Figure(go.Scatter(x=Eastern['Date'],

               y=Eastern['Hydro Generation (in %)'],

               marker_color='rgb(0, 0, 204)',

               name='Eastern Region')

)

fig_four.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_four.update_layout(yaxis=dict(range=[-50,50]), title_text='Eastern Region')

fig_four.show()



fig_five = go.Figure(go.Scatter(x=NE['Date'],

               y=NE['Hydro Generation (in %)'],

               marker_color='rgb(0, 0, 204)',

               name='NE Region')

)

fig_five.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(

                    buttons=list([

                    dict(count=1, label="1m", step="month", stepmode="backward"),

                    dict(count=6, label="6m", step="month", stepmode="backward"),

                    dict(count=1, label="YTD", step="year", stepmode="todate"),

                    dict(count=1, label="1y", step="year", stepmode="backward"),

                    dict(step="all")

                    ])

                                 )

                )

fig_five.update_layout(yaxis=dict(range=[-50,50]), title_text='NE Region')

fig_five.show()