import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

%matplotlib inline





# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Kaggle.csv")

df.head()
df.count()
data = dict(type = 'choropleth', 

           locations = df['Id'],

           locationmode = 'country names',

           z = df['Human Development Index HDI-2014'], 

           text = df['Id'],

           colorbar = {'title':'Human development index'}

            

           )

layout = dict(title = 'human development index', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

fig = go.Figure(data = [data], layout=layout)

iplot(fig, validate=False, filename='d3-world-map' )





data = dict(type = 'choropleth', 

           locations = df['Id'],

           locationmode = 'country names',

           z = df['Gini coefficient 2005-2013'], 

           text = df['Id'],

           colorbar = {'title':'Gini index'}

            

           )

layout = dict(title = 'Gini index', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

fig = go.Figure(data = [data], layout=layout)

iplot(fig, validate=False, filename='d3-world-map' )


