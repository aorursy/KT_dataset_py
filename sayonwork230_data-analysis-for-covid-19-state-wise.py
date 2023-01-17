!pip install plotly
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.offline as plo

import plotly.graph_objects as go

from plotly.subplots import make_subplots
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv', index_col = 'Date')

df.index = pd.to_datetime(df.index, format="%d/%m/%y")

df = df.drop(['Sno'], axis = 1)

df.head()
df_details = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')

df_details.head()
df_population = pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv')

df_population.head()
df_hospital = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')

df_hospital.head()
infodf = pd.pivot_table(df, values=['Confirmed','Deaths','Cured'], index='State/UnionTerritory', aggfunc='max')
infodf = infodf.sort_values(by='Confirmed', ascending= False)

infodf.style.background_gradient(cmap='Wistia')
print("Total Cases in India:",int(infodf['Confirmed'].sum()))
data = [go.Bar(

    x = infodf.index,

    y = infodf[colname],

    name = colname

)for colname in infodf.columns]



fig = go.Figure(data=data)



plo.iplot(fig)
lockdown_date = '2020-03-25'
def plotly_graph_state(state):

    temp = df[df['State/UnionTerritory']==state]

    trace0 = go.Scatter(

        x = temp.index,

        y = temp['Confirmed'],

        mode = 'lines+markers',

        marker = dict(color='green'),

        name = 'Confirmed Cases in {0}'.format(state)

    )

    layout = go.Layout(

        title = 'Confirmed Cases in {0}'.format(state)

    )

    fig = go.Figure(

        data = [trace0],

        layout = layout

    )

    plo.iplot(fig)
all_states = list(infodf.sort_values(by='Confirmed', ascending= False).index[:5])

for state in all_states:

    plotly_graph_state(state)
def plotly_facts(state):

    fig = make_subplots(rows=2, cols=2, specs=[

        [{"type": "bar"}, {"type": "pie"}],

        [{"type": "bar"}, {"type": "bar"}]

    ], subplot_titles=("Status of Patients","Gender Ratio", "Number of Hospitals", "Population"))

    temp = df_details[df_details['detected_state']==state]

    trace1 = go.Bar(

        x = temp['current_status'].value_counts().index,

        y = temp['current_status'].value_counts().values,

        name = 'Status of People'

    )

    trace2 = go.Pie(

        labels=['Male','Female'],

        values=temp['gender'].dropna().value_counts(),

        showlegend=False

    )

    tt1 = df_hospital[df_hospital['State/UT']==state].iloc[:,[8,10]]

    values = [int(tt1.loc[:,colname].values) for colname in tt1.columns]

    trace3 = go.Bar(

        x = tt1.columns,

        y = values,

    )



    tt2 = df_population[df_population['State / Union Territory']==state].iloc[:,[3,4]]

    values = [int(tt2.loc[:,colname].values) for colname in tt2.columns]

    trace4 = go.Bar(

        x = tt2.columns,

        y = values,  

    )

    

    fig.add_trace(trace1, row=1, col=1)

    fig.add_trace(trace2, row=1, col=2)

    fig.add_trace(trace3, row=2, col=1)

    fig.add_trace(trace4, row=2, col=2)

    fig.update_layout(height=1000, showlegend=False,title_text="Situation in {0}".format(state))

    plo.iplot(fig)
select_states = ['Kerala', 'Maharashtra', 'Delhi', 'Karnataka']

for state in select_states:

    plotly_facts(state)