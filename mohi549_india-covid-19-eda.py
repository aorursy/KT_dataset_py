import numpy as np 

import pandas as pd 

import os

from plotly import tools

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go

import plotly.express as px

import warnings

warnings.filterwarnings("ignore")

init_notebook_mode(connected=True)

pd.set_option('display.max_columns', None)

print(os.listdir("../input"))

pd.set_option('display.max_columns', None)

import operator

import numpy

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

from collections import Counter

from wordcloud import WordCloud           ## To Generate Wordcloud

from datetime import datetime             ## Work with timeseries data

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))


individual = pd.read_csv("/kaggle/input/covid19-in-india/IndividualDetails.csv")

s_testing= pd.read_csv("/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv")

hospital = pd.read_csv("/kaggle/input/covid19-in-india/HospitalBedsIndia.csv")

india = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")

agegrp = pd.read_csv("/kaggle/input/covid19-in-india/AgeGroupDetails.csv")

testinglab = pd.read_csv("/kaggle/input/covid19-in-india/ICMRTestingLabs.csv")

census = pd.read_csv("/kaggle/input/covid19-in-india/population_india_census2011.csv")



print("data Shape:", individual.shape)
s_testing = s_testing.fillna(0)

s_testing['Negative'] =pd.to_numeric(s_testing['Negative'],errors = 'coerce')

sumup = s_testing.groupby(['State'])['TotalSamples','Negative','Positive'].agg({'TotalSamples':'max','Negative':'max','Positive':'max'}).sort_values(by =['TotalSamples','Positive','Negative'],ascending = False).reset_index()

sumup['unknown'] = sumup['TotalSamples']-(sumup['Negative']+sumup['Positive'])

sumup['Negative(%)'] = round((sumup['Negative']/sumup['TotalSamples'])*100,1)

sumup['Positive(%)'] = round((sumup['Positive']/sumup['TotalSamples'])*100,1)

sumup['unknown(%)'] = round((sumup['unknown']/sumup['TotalSamples'])*100,1)



fig = make_subplots(

    rows=3, cols=1,

    shared_xaxes=True,

    vertical_spacing=0.03,

    specs=[[{"type": "table"}],

           [{"type": "Bar"}],

           [{"type": "Bar"}]]

)



fig.add_trace(

    go.Bar(

        y=sumup["Positive"],

        x=sumup["State"],

        marker=dict(color="#DB4437"),

        name="Positive Cases state wise"

    ),

    row=3, col=1

)



fig.add_trace(

    go.Bar(

        y=sumup["Negative"],

        x=sumup["State"],

        marker=dict(color="#4285F4"),

        name="Negative Cases state wise"

    ),

    row=2, col=1

)



fig.add_trace(

    go.Table( columnwidth = [40] + [33, 35, 33],

    header=dict(values=list(sumup.columns),

                fill_color='Orange',font=dict(size=10),align="left"),

    cells=dict(values=[sumup.State,sumup.TotalSamples,sumup.Negative,sumup.Positive,sumup.unknown,sumup['Positive(%)'],sumup['Negative(%)'],sumup['unknown(%)']],

               fill_color='white',

               align='left')

),

    row=1,col=1

    )



fig.update_layout(

    height=800,

    showlegend=False,

    title_text="Corona Testing by States",

)



fig.show()
state_hospital = hospital[hospital['State/UT']!="All India"]

top_20_u=state_hospital.nlargest(20,'NumUrbanHospitals_NHP18')

top_20_r=state_hospital.nlargest(20,'NumRuralHospitals_NHP18')





trace1 = go.Bar(x=top_20_u["NumUrbanHospitals_NHP18"], y=top_20_u['State/UT'], orientation='h', name="Urban Hospitals", marker=dict(color='#F4B400'))

trace2 = go.Bar(x=top_20_r["NumRuralHospitals_NHP18"], y=top_20_r['State/UT'], orientation='h', name="Rural Hospitals", marker=dict(color='#0F9D58'))



data = [trace1,trace2]

layout = go.Layout(title="Urban and Rural Hospitals By States", legend=dict(x=0.1, y=1.1, orientation="h"),plot_bgcolor='rgba(0,0,0,0)')

fig = go.Figure(data, layout=layout)

fig.show()
state_hospital = hospital[hospital['State/UT']!="All India"]

state_hospital['NumPrimaryHealthCenters_HMIS'] = pd.to_numeric(state_hospital['NumPrimaryHealthCenters_HMIS'],errors='coerce')

top_20_u=state_hospital.nlargest(20,'NumPrimaryHealthCenters_HMIS')

top_20_r=state_hospital.nlargest(20,'NumCommunityHealthCenters_HMIS')





trace1 = go.Bar(x=top_20_u["NumPrimaryHealthCenters_HMIS"], y=top_20_u['State/UT'], orientation='h', name="Urban Hospitals", marker=dict(color='#4285F4'))

trace2 = go.Bar(x=top_20_r["NumCommunityHealthCenters_HMIS"], y=top_20_r['State/UT'], orientation='h', name="Rural Hospitals", marker=dict(color='#DB4437'))



data = [trace1,trace2]

layout = go.Layout(title="Primary and Community Hospitals by state", legend=dict(x=0.1, y=1.1, orientation="h"),plot_bgcolor='rgba(0,0,0,0)')

fig = go.Figure(data, layout=layout)

fig.show()
hospital = hospital.fillna(0)

state_hospital = hospital[hospital['State/UT']!="All India"]

ind=['NumUrbanBeds_NHP18', 'NumRuralBeds_NHP18', 'NumPublicBeds_HMIS']



fig = go.Figure(data=[

    go.Bar(name='Public Beds', x=state_hospital["State/UT"], y=state_hospital.NumPublicBeds_HMIS),

    go.Bar(name='Rural Beds', x=state_hospital["State/UT"], y=state_hospital.NumRuralBeds_NHP18),

    go.Bar(name='Urban Beds', x=state_hospital["State/UT"], y=state_hospital.NumUrbanBeds_NHP18)

    

## plot

])



fig.update_layout(

    height=500,

    showlegend=False,barmode='stack',

    title_text="Beds in Hospitals by States in india",

)

fig.show()

# for col in hospital.columns[2:]:

#     if hospital[col].dtype=='object':

#         hospital[col]=hospital[col].astype('float')


ind=['NumUrbanBeds_NHP18', 'NumRuralBeds_NHP18', 'NumPublicBeds_HMIS']

    

## plot

trace = go.Pie(labels=ind, values=[431173,279588,739024], pull=[0.05, 0], marker=dict(colors=['#F4B400','#0F9D58','#4285F4']))

layout = go.Layout(title="Overall Beds available in India", height=400, legend=dict(x=1

                                                                                          , y=1.1))

fig = go.Figure(data = [trace], layout = layout)

iplot(fig)
col = "type"

grouped = testinglab[col].value_counts().reset_index()

grouped = grouped.rename(columns = {col : "count", "index" : col})



## plot

trace = go.Pie(labels=grouped[col], values=grouped['count'],  hole=.3,pull=[0.05, 0], marker=dict(colors=['#F4B400','#0F9D58','#4285F4']))

layout = go.Layout(title="Types of Lab Testing Centers in India", height=400, legend=dict(x=1

                                                                                          , y=1.1))

fig = go.Figure(data = [trace], layout = layout)

iplot(fig)
govlab= testinglab[testinglab['type']=="Government Laboratory"]

colsite = testinglab[testinglab['type']=="Collection Site"]

pvtlab = testinglab[testinglab['type']=="Private Laboratory"]



col = "state"



vc1 = govlab[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))



vc2 = colsite[col].value_counts().reset_index()

vc2 = vc2.rename(columns = {col : "count", "index" : col})

vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))



vc3 = pvtlab[col].value_counts().reset_index()

vc3 = vc3.rename(columns = {col : "count", "index" : col})

vc3['percent'] = vc3['count'].apply(lambda x : 100*x/sum(vc3['count']))



trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="Government Laboratory", marker=dict(color='#4285F4'))

# trace2 = go.Bar(x=vc2[col], y=vc2["count"], name="Collection Site", marker=dict(color='#DB4437'))

trace3 = go.Bar(x=vc3[col], y=vc3["count"], name="Private Laboratory", marker=dict(color='#DB4439'))

data = [trace1,trace3]

layout = go.Layout(title="Test Centers by states", legend=dict(x=0.1, y=1.1, orientation="h"),plot_bgcolor='rgba(0,0,0,0)')

fig = go.Figure(data, layout=layout)

fig.show()
fig = go.Figure(data=[go.Table(

    header=dict(values=list(["City","Address"]),

                fill_color='red',

                align='left'),

    cells=dict(values=[pvtlab.city,pvtlab.lab],

               fill_color='white',

               align='left',line_color='darkslategray'))

])



fig.update_layout(

    height=400,

    showlegend=False,

    title_text="Private Testing Centers In India",

)

fig.show()
fig = go.Figure(data=[go.Table(

    header=dict(values=list(["City","Address"]),

                fill_color='green',

                align='left'),

    cells=dict(values=[govlab.city,govlab.lab],

               fill_color='white',

               align='left',line_color='darkslategray'))

])



fig.update_layout(

    height=400,

    showlegend=False,

    title_text="Government Testing Centers In India",

)

fig.show()