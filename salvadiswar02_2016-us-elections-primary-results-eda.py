# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import io
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode()

import matplotlib.pyplot as plt

import seaborn as sns

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
primary_results =  pd.read_csv('../input/primary_results.csv')

new_df = primary_results.groupby(['state','state_abbreviation','party']).fraction_votes.mean()

new_df = pd.DataFrame(new_df)

df = new_df.reset_index()
result = df

scl = [[0, 'rgb(0,100,0)'],[1, 'rgb(0,191,255)']]

#result['text'] = result['state'] + '<br>' +\
#'Democrat '+result['Democrat'].astype(str)+' Republican '+result['Republican'].astype(str)


    
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = result['state_abbreviation'],
        z = result['fraction_votes'].astype(float),
        locationmode = 'USA-states',
        #text = result['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Votes Percentage")
        ) ]

layout = dict(
        title = '2016 US ELECTIONS Trend between Democrat and Republican Party',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )

new_list = primary_results.groupby(['state','state_abbreviation','candidate']).fraction_votes.mean()

candidates_list = ['state_abbreviation','candidate','fraction_votes']

new_list = pd.DataFrame(new_list)

new_list = new_list.reset_index()

new_list = new_list[candidates_list] 

new_list = new_list.loc[new_list.candidate.isin(['Bernie Sanders','Donald Trump','Hillary Clinton'])]

new_list = new_list.reset_index()

election_data = new_list.pivot_table(index = 'state_abbreviation',columns ='candidate' , values = 'fraction_votes' )

election_data = pd.DataFrame(election_data)

#election_data = election_data.reset_index()

columns_new = ['Bernie Sanders','Donald Trump','Hillary Clinton']


sns.set(rc={'figure.figsize':(5,20)})

sns.heatmap(election_data[columns_new],cmap='YlGnBu',xticklabels=True, yticklabels=True)