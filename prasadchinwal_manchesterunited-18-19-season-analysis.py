# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import plotly as py

import plotly.graph_objs as go

import ipywidgets as widgets

from scipy import special

import plotly.express as px

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Set plotly for offline mode

py.offline.init_notebook_mode(connected=True)



import os

data = pd.read_excel('/kaggle/input/manchester-united-season-201819/Manchester United 2018-19.xlsx')

# Rename Columns

data = data.rename(columns={"Unnamed: 1": "opponent"})

data = data.rename(columns={"Unnamed: 2": "goals_scored"})

data = data.rename(columns={"Unnamed: 3": "goals_taken"})

data = data.rename(columns={"Unnamed: 4": "result"})

data = data.rename(columns={"Unnamed: 5": "possession"})

data = data.rename(columns={"Unnamed: 9": "location"})
# Select required column data

data = data[["opponent", "goals_scored", "goals_taken", "result", "possession", "location"]]

data = data.drop([0],axis=0)
fig = px.scatter(data, x='opponent', y='result')

fig.show()
# When possession greater than opponent

dataMorePossession = data[data['possession'] > 50]

fig2 = px.scatter(dataMorePossession, x='opponent', y='result')

fig2.show()
# When playing home vs away

fig3 = px.line(data, x='location', y='result')

fig3.show()