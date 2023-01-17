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
import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot
def load_data():

    data = pd.read_csv(DATA_URL)

    data = data.drop(data.index[0])

    data['Start Date'] = pd.to_datetime(data['Start Date'])

    data['New Displacements'] = data['New Displacements'].fillna(0).astype(int)

    data['Year'] = data['Year'].fillna(0).astype(int)

    return data
DATA_URL = "../input/idmc-internally-displaced-persons-data/disaster_data.csv"

data = load_data()
# Only filter for South East Asian countries

SEA_countries = ['Brunei', 'Cambodia', 'East Timor', 'Indonesia', 'Laos', 'Malaysia', 'Myanmar', 'Philippines', 'Indonesia', 'Singapore', 'Thailand', 'Vietnam']

SEA_data = data[data['Name'].isin(SEA_countries)]
# 1.A) Which Hazard Type is the responsible for the largest number of displacements?

# 1.B) Which is the most common Hazard Type based on the total count of occurrences from 2008-2019? 

common_hazard_type = pd.DataFrame(SEA_data.groupby('Hazard Type')['New Displacements'].sum()).sort_values('New Displacements', ascending=False).reset_index().rename(columns={'New Displacements': 'Interally Displaced Persons'})

top_hazard_type = pd.DataFrame(SEA_data.groupby('Hazard Type')['New Displacements'].count()).sort_values('New Displacements', ascending=False).reset_index().rename(columns={'New Displacements': 'Number of disasters'})

scatter_data = pd.merge(common_hazard_type, top_hazard_type, on=['Hazard Type'], how='left')



fig = px.scatter(

  scatter_data, 

  title="Number of Internally Displaced Persons vs Number of Disaster Occurrences",

  y='Number of disasters',

  x='Interally Displaced Persons',

  log_x=True,

  height=600,

  text='Hazard Type'

)



fig.update_traces(textposition='top center', marker=dict(size=10))



fig.update_layout(font_family="Courier New", template="plotly_dark")



iplot(fig)
# 2) Which country is the most affected by displacements based on A) total IDPs



SEA_data_2 = SEA_data[

  SEA_data['Name'].isin(['Philippines', 'Indonesia', 'Myanmar', 'Thailand', 'Vietnam'])

]



updatemenus = list([

    dict(active=1,

         buttons=list([

            dict(label='Log Scale',

                 method='update',

                 args=[{'visible': [True, True]},

                       {'yaxis': {'type': 'log'}}]),

            dict(label='Linear Scale',

                 method='update',

                 args=[{'visible': [True, False]},

                       {'yaxis': {'type': 'linear'}}])

            ]),

        )

    ])





top_country_idps = pd.DataFrame(SEA_data_2.groupby(['Year', 'Name'])['New Displacements'].sum()).sort_values('New Displacements', ascending=True).reset_index().sort_values(['Year', 'Name'], ascending=True)



trace1 = px.line(

  top_country_idps, 

  y='New Displacements',

  x='Year',

  color='Name',

  width=1000,

  height=600

)



trace1.update_traces(mode='lines+markers', connectgaps=True)



layout = dict(updatemenus=updatemenus, 

              font_family="Courier New", 

              template="plotly_dark",

              title="Total Internally Displaced Persons from 2008-2019", 

              height=600,

              legend=dict(

                  orientation="h", y=1.03, yanchor="bottom", x=0.5, xanchor="center", borderwidth=1

              )

             )

fig = go.Figure(data=trace1.data, layout=layout)

iplot(fig)
# 2) Which country is the most affected by displacements based on B) total hazard occurrences

top_country_idps = pd.DataFrame(SEA_data_2.groupby(['Year', 'Name'])['New Displacements'].count()).sort_values('New Displacements', ascending=True).reset_index().sort_values(['Year', 'Name'], ascending=True)



trace2 = px.line(

  top_country_idps, 

  y='New Displacements',

  x='Year',

  color='Name',

)



trace2.update_traces(mode='lines+markers', connectgaps=True)



layout = dict(updatemenus=updatemenus, 

              font_family="Courier New", 

              template="plotly_dark",

              title="Total Disaster Occurrences from 2008-2019", 

              height=600,

              legend=dict(

                  orientation="h", y=1.03, yanchor="bottom", x=0.5, xanchor="center", borderwidth=1

              ))

fig = go.Figure(data=trace2.data, layout=layout)

iplot(fig)
# 2C) Bubble chart showing the total number of displacements vs total disaster occurrence per country from 2008-2019



SEA_data_1 = SEA_data[

  SEA_data['Hazard Type'].isin(['Flood', 'Storm']) &

  SEA_data['Year'].isin(range(2008, 2020)) &

  SEA_data['Name'].isin(['Philippines', 'Indonesia', 'Myanmar', 'Thailand', 'Vietnam'])

]



type_sum_by_year = pd.DataFrame(SEA_data_1.groupby(['Year', 'Hazard Type', 'Name'])['New Displacements'].sum()).reset_index().rename(columns={'New Displacements': 'Interally Displaced Persons'})

type_count_by_year = pd.DataFrame(SEA_data_1.groupby(['Year', 'Hazard Type', 'Name'])['New Displacements'].count()).reset_index().rename(columns={'New Displacements': 'Number of disasters'})

bar_data = pd.merge(type_sum_by_year, type_count_by_year, on=['Year', 'Hazard Type', 'Name'], how='left').sort_values(['Year','Name'], ascending=True)



sizeref = 2.*max(bar_data['Number of disasters'])/(100**2)



fig = px.scatter(

  bar_data, 

  title="Number of Interally Displaced Persons by Floods and Storms from 2008 - 2019",

  size='Number of disasters',

  x='Year',

  y='Interally Displaced Persons',

  log_y=True,

  color='Name',

  facet_col='Hazard Type'

)



fig.update_traces(mode='markers', marker=dict(sizeref=sizeref, line_width=1))



fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=14)))



fig.update_layout( # customize font and legend orientation & position

  font_family="Courier New",

  template="plotly_dark",

  grid=dict(xgap= 0.5),

  margin=dict(pad=5),

  title=dict(font=dict(size=18), y=0.98),

  height=900,

  legend=dict(

      orientation="h", y=1.03, yanchor="bottom", x=0.5, xanchor="center", borderwidth=1

  )

)



iplot(fig)