# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plot graphs and requirement for pandas_profiling
import pandas_profiling #the complete report for a dataset
from plotly import __version__ #version of plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot #plotly offline
init_notebook_mode(connected=True) #plotly offline
import plotly.graph_objs as go #plotly 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/crisis-data.csv')
pandas_profiling.ProfileReport(df)
print("Plotly version: {}".format(__version__)) #requires version >= 1.9.0
df1 = df[['Template ID','Call Type']].groupby(['Call Type']).count().reset_index()
df1 = df1.sort_values(['Template ID'], ascending=False)

data = [go.Bar(
            y=df1['Call Type'],
            x=df1['Template ID'],
            orientation = 'h'
    )]


layout = go.Layout(    
    title='Call Type Count',
    margin=dict(
        l=220,
        r=10,
        t=140,
        b=80
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)

df_ = df[['Template ID','Precinct']]
df_.drop_duplicates(inplace=True)
df2 = df_[['Template ID','Precinct']].groupby(['Precinct']).count().reset_index()
df2 = df2.sort_values(['Template ID'], ascending=False)

data = [go.Bar(
            y=df2['Precinct'][:10],
            x=df2['Template ID'][:10],
            orientation = 'h'
    )]


layout = go.Layout(
    title='Precinct Count (Without duplicate values)',  
    margin=dict(
        l=120,
        r=10,
        t=140,
        b=80
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)



