# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objects as go

import plotly as py

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
supplyFood  = pd.read_csv("/kaggle/input/covid19-healthy-diet-dataset/Supply_Food_Data_Descriptions.csv")

fatSupply = pd.read_csv("/kaggle/input/covid19-healthy-diet-dataset/Fat_Supply_Quantity_Data.csv")

supplyQuantity = pd.read_csv("/kaggle/input/covid19-healthy-diet-dataset/Food_Supply_Quantity_kg_Data.csv")

kcal = pd.read_csv("/kaggle/input/covid19-healthy-diet-dataset/Food_Supply_kcal_Data.csv")

protein = pd.read_csv("/kaggle/input/covid19-healthy-diet-dataset/Protein_Supply_Quantity_Data.csv")
for i in range(1,22):

    fig = px.bar(x = protein.iloc[protein.iloc[:,i].sort_values(ascending = False)[:3].reset_index()['index'],0],y = protein.iloc[:,i].sort_values(ascending = False)[:3],

                 color =protein.iloc[protein.iloc[:,i].sort_values(ascending = False)[:3].reset_index()['index'],0], height =300)

    fig.update_layout(title_text= protein.columns[i])

    fig.show()
india  = protein[protein['Country']=='India'].iloc[:,1:22]

fig = go.Figure(

    data=[

        go.Pie(labels=india.T.index.tolist(), values=india.iloc[0,:].tolist())

    ],

    layout=go.Layout(

        title="Category distribution in India",

       

    )

)



fig.show()

maxProteinCategory = []

for i in range(1,22):

    maxProteinCategory.append(protein.iloc[:,i].sort_values(ascending = False)[0])

Max = max(maxProteinCategory)

Index = maxProteinCategory.index(Max) +1



data = dict (

    type = 'choropleth',

    locations = protein['Country'],

    locationmode='country names',

    z=protein[protein.columns[Index]])

go.Figure(data=[data])

active = (protein['Active']*protein['Population'])/100

death = (protein['Deaths']*protein['Population'])/100

recovered = (protein['Recovered']*protein['Population'])/100
fig = go.Figure(

    data=[

        go.Bar(

            name="Active",

            x=protein.iloc[active.sort_values(ascending = False)[:15].reset_index()['index'],0],

            y=active.sort_values(ascending = False)[:15],

            offsetgroup=0,

        ),

    ],

    layout=go.Layout(

        title="Top 15 Countries with higher number of active case",

        yaxis_title="Number of Cases"

    )

)

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
fig2 = go.Figure(

    data = [

        go.Bar(

            name="Death",

            x=protein.iloc[death.sort_values(ascending = False)[:15].reset_index()['index'],0],

            y=death.sort_values(ascending = False)[:15],

            offsetgroup=1,

        )

    ],

    layout=go.Layout(

        title="Top 15 Counteries with higher number of death",

    )

)

fig2.show()
fig3 = go.Figure(

    data = [

        go.Bar(

            name="Recovered",

            x=protein.iloc[recovered.sort_values(ascending = False)[:15].reset_index()['index'],0],

            y=recovered.sort_values(ascending = False)[:15],

            offsetgroup=1,

        )

    ],

    layout=go.Layout(

        title="Top 15 Counteries with higher number of recovered",

    )

)

fig3.show()
india  = kcal[kcal['Country']=='India'].iloc[:,1:22]

fig = go.Figure(

    data=[

        go.Pie(labels=india.T.index.tolist(), values=india.iloc[0,:].tolist())

    ],

    layout=go.Layout(

        title="kCal distribution in India",

       

    )

)



fig.show()

india  = fatSupply[fatSupply['Country']=='India'].iloc[:,1:22]

fig = go.Figure(

    data=[

        go.Pie(labels=india.T.index.tolist(), values=india.iloc[0,:].tolist())

    ],

    layout=go.Layout(

        title="Fat supply distribution in India",

       

    )

)



fig.show()

india  = supplyQuantity[supplyQuantity['Country']=='India'].iloc[:,1:22]

fig = go.Figure(

    data=[

        go.Pie(labels=india.T.index.tolist(), values=india.iloc[0,:].tolist())

    ],

    layout=go.Layout(

        title="Supply quantity distribution in India",

       

    )

)



fig.show()
