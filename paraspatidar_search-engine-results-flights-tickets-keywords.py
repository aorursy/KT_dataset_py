# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/serp_flights.csv")

data.head()
data.describe()
import missingno as msno

msno.matrix(data)

plt.show()
data.isnull().sum()
new_data = data[['searchTerms','title','snippet','displayLink','queryTime','formattedSearchTime']]

new_data.head()
import missingno as msno

msno.matrix(new_data)

plt.show()
new_data.isnull().sum()
websites_count = data['displayLink'].value_counts()

websites_count
new_data['WebsiteCounts'] = new_data.groupby(['displayLink'])['formattedSearchTime'].transform('count')

new_data.head()
new_data = new_data.rename(columns={'displayLink':'Website'})

new_data.head()
from plotly import tools

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

from IPython.display import HTML, Image
colors = {

    "Bug": "#A6B91A",

    "Dark": "#705746",

    "Dragon": "#6F35FC",

    "Electric": "#F7D02C",

    "Fairy": "#D685AD",

    "Fighting": "#C22E28",

    "Fire": "#EE8130",

    "Flying": "#A98FF3",

    "Ghost": "#735797",

    "Grass": "#7AC74C",

    "Ground": "#E2BF65",

    "Ice": "#96D9D6",

    "Normal": "#A8A77A",

    "Poison": "#A33EA1",

    "Psychic": "#F95587",

    "Rock": "#B6A136",

    "Steel": "#B7B7CE",

    "Water": "#6390F0",

}
data = go.Bar(

    x=new_data['Website'][:20],

    y=new_data['WebsiteCounts'][:20],

    marker=dict(

        color=list(colors.values())

    ),



)



layout = go.Layout(

    title='Webiste Counts',

    xaxis=dict(

        title='Webistes'

    ),

    yaxis=dict(

        title='Webistes Count'

    )

)



fig = go.Figure(data=[data], layout=layout)

iplot(fig, filename='Webiste Counts')
new_data['title'].value_counts()
new_data['searchTerms'].value_counts()