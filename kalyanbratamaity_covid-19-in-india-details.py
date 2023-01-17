# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# show all the cols and rows of a dataframe

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option('display.width', 1000)



# for visualization

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.orca.config.use_xvfb = False

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
covid_india = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

print(covid_india.shape)

covid_india.head()
# list of dates and states list

dates = pd.DataFrame(covid_india.Date.unique().tolist(), columns=['Date'])

states = pd.DataFrame(covid_india['State/UnionTerritory'].unique().tolist(), columns=['States'])



dates['key'] = 0

states['key'] = 0



crossed = pd.merge(dates, states, on='key')
covid_india_new = pd.merge(crossed, covid_india, 

                           left_on=['Date', 'States'],

                           right_on=['Date', 'State/UnionTerritory'],

                           how='left')



covid_india_new.Date=pd.to_datetime(covid_india_new.Date, dayfirst=True)
covid_india_new = covid_india_new.groupby(['States'], as_index=False).apply(lambda group: group.ffill())

covid_india_new = covid_india_new.groupby(['States'], as_index=False).apply(lambda group: group.fillna(0))
covid_india_new['TotalConfirmed'] = covid_india_new['ConfirmedIndianNational']+covid_india_new['ConfirmedForeignNational']
covid_india_new.drop(['State/UnionTerritory', 'Sno', 'key'], axis=1, inplace=True)

covid_india_new.rename(columns={'States' : 'State/UnionTerritory'}, inplace=True)
covid_india_new.head(5)
for state in covid_india['State/UnionTerritory'].unique().tolist():

    

    # for each of the state

    state_data = covid_india_new[covid_india_new['State/UnionTerritory'] == state]

    state_grpd = state_data.groupby('Date')['Date', 'TotalConfirmed', 'Cured', 'Deaths'].sum().reset_index()



    # plot

    fig = go.Figure(layout=dict(title=dict(text=state)))



    fig.add_trace(go.Scatter(x=state_grpd['Date'], y=state_grpd['TotalConfirmed'], mode='lines+markers', name='Confirmed Cases'))

    fig.add_trace(go.Scatter(x=state_grpd['Date'], y=state_grpd['Cured'], mode='lines+markers', name='Cured'))

    fig.add_trace(go.Scatter(x=state_grpd['Date'], y=state_grpd['Deaths'], mode='lines+markers', name='Deaths'))

        

    fig.show()