# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



train1 = pd.read_csv("../input/name_geographic_information.csv")

train2 = pd.read_csv("../input/base_etablissement_par_tranche_effectif.csv")

#train1.head()

train2.head()

train1.describe()
train2.describe()
train1.isnull().any().any()
train1.isnull().sum()
train2=train2.drop(train2[train2['CODGEO'].apply(lambda x: str(x).isdigit())==False].index)



train2['CODGEO']=train2['CODGEO'].astype(int)

train2.describe()
train=pd.merge(train1,train2,left_on='code_insee',right_on='CODGEO')

train.head()
train_paris=train[train['LIBGEO']=='Paris']

train_paris.describe()

col_lst=['E14TST', 'E14TS0ND', 'E14TS1', 'E14TS6', 'E14TS10', 'E14TS20',

       'E14TS50', 'E14TS100', 'E14TS200', 'E14TS500']

data = [go.Bar(

            x = train_paris[col_lst].columns.values,

            y = train_paris[col_lst].sum()/train_paris[col_lst].count()

    )]



layout = go.Layout(

    title='Paris number of industries by size group'

)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='basic-bar')
data = [go.Bar(

            x = train['nom_région'],

            y = train['E14TS500']

    )]



layout = go.Layout(

    title='Regions number of big industries (>500)'

)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='basic-bar')


#trace0= go.Bar(

#    x = train['nom_région'],

#    y = train['E14TS500']

#)

#trace1 = go.Bar(

#    x = train['nom_région'],

#    y = train['E14TS200']

#)

#trace2 = go.Bar(

#    x = train['nom_région'],

#    y = train['E14TS100']

#)

#data = [trace0,trace1,trace2]

#layout = go.Layout(barmode='stack')

#fig = go.Figure(data=data, layout=layout)



#py.iplot(fig, filename='stacked histogram for industries by region')