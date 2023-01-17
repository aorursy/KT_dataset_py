# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print("Reading in the data files")
matches = pd.read_csv("../input/matches.csv")
deliveries = pd.read_csv("../input/deliveries.csv")
print("Head of the matches dataframe")
matches.head()
print("Summary stats for continuous features in matches data")
matches.describe()
print("Head of deliveries data")
deliveries.head()
print("Summary statistics for the continuous features in the deliveries data")
deliveries.describe()
print("shape of DELIVERIES df")
deliveries.shape
print("Importing the libraries required to generate plots\nWe will use plotly")
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import squarify
import matplotlib
init_notebook_mode(connected = True)
print("Creating the data for a treemap \nWe want to see the top5 bowlers who given the most runs off the bat")

offTheBat = deliveries.groupby(['bowler']).sum()['batsman_runs'].reset_index().sort_values(by = ['batsman_runs'], ascending=False)
offTheBat_top20 = offTheBat.iloc[0:21,:]
offTheBat_top20 = offTheBat_top20.sort_values(by = ['batsman_runs'])
print("Creating the treemap \nTHE COSTLIEST BOWLERS W.R.T RUNS OFF THE BAT (NO EXTRAS)")
x = 0.
y = 0.
width = 100.
height = 100.

values = list(offTheBat_top20.batsman_runs)
bowlers_1 = list(offTheBat_top20.bowler)

'''Takes in the treemap value list, width, height and throws out
A list of positive values sorted from largest to smallest and normalized to
the total area
'''
normed = squarify.normalize_sizes(values, width, height)

'''The function returns a list of JSON objects, each one a rectangle with
coordinates corresponding to the given coordinate system and area proportional
to the corresponding value.'''
rects = squarify.squarify(normed, x, y, width, height)

# Choose colors from http://colorbrewer2.org/ under "Export"
color_brewer = ['rgb(255,255,255)','rgb(230,230,255)','rgb(204,204,255)',
                'rgb(179,179,255)','rgb(153,153,255)','rgb(128,128,255)',
               'rgb(102,102,255)','rgb(77,77,255)','rgb(51,51,255)',
               'rgb(26,26,255)','rgb(0,0,255)','rgb(0,0,230)',
               'rgb(0,0,204)','rgb(0,0,179)','rgb(0,0,153)',
               'rgb(0,0,128)','rgb(0,0,102)','rgb(0,0,77)',
               'rgb(0,0,51)','rgb(0,0,26)','rgb(0,0,0)']
shapes = []
annotations = []
counter = 0

'''
The following loop creates subdicts taking the above information.
1 dictionary each is made for each treemap rectangle specifying 
additional properties like line, color and type = 'rect'
'''
for r in rects:
    shapes.append( 
        dict(
            type = 'rect', 
            x0 = r['x'], 
            y0 = r['y'], 
            x1 = r['x']+r['dx'], 
            y1 = r['y']+r['dy'],
            line = dict( width = 2 ),
            fillcolor = color_brewer[counter]
        ) 
    )
    annotations.append(
        dict(
            x = r['x']+(r['dx']/2),
            y = r['y']+(r['dy']/2),
            text = bowlers_1[counter], 
            showarrow = False
        )
    )
    counter = counter + 1
    if counter >= len(color_brewer):
        counter = 0

        
# For hover text
trace0 = go.Scatter(
    x = [ r['x']+(r['dx']/2) for r in rects ], 
    y = [ r['y']+(r['dy']/2) for r in rects ],
    text = values,
    mode = 'text',
)
        
layout = dict(
    height=700, 
    width=700,
    xaxis=dict(showgrid=False,zeroline=False),
    yaxis=dict(showgrid=False,zeroline=False),
    shapes=shapes,
    annotations=annotations,
    hovermode='closest'
)

# With hovertext
figure = dict(data=[trace0], layout=layout)

# Without hovertext
# figure = dict(data=[Scatter()], layout=layout)

iplot(figure, filename='squarify-treemap')


del_noballs = deliveries[deliveries['noball_runs'] > 0]
del_noballs = del_noballs[['bowler', 'noball_runs', 'bye_runs']]
del_noballs = del_noballs.groupby(['bowler']).count()['noball_runs'].reset_index().sort_values(['noball_runs'], ascending = False)
del_noballs_top = del_noballs.iloc[0:21,:]

noballs = del_noballs_top.sort_values(['noball_runs'])

print("Creating the treemap \nBOWLERS with MAX NUMBER OF NOBALLS")
x = 0.
y = 0.
width = 100.
height = 100.

values = list(noballs.noball_runs)
bowlers_1 = list(noballs.bowler)

'''Takes in the treemap value list, width, height and throws out
A list of positive values sorted from largest to smallest and normalized to
the total area
'''
normed = squarify.normalize_sizes(values, width, height)

'''The function returns a list of JSON objects, each one a rectangle with
coordinates corresponding to the given coordinate system and area proportional
to the corresponding value.'''
rects = squarify.squarify(normed, x, y, width, height)

# Choose colors from http://colorbrewer2.org/ under "Export"
color_brewer = ['rgb(255,255,255)','rgb(230,230,255)','rgb(204,204,255)',
                'rgb(179,179,255)','rgb(153,153,255)','rgb(128,128,255)',
               'rgb(102,102,255)','rgb(77,77,255)','rgb(51,51,255)',
               'rgb(26,26,255)','rgb(0,0,255)','rgb(0,0,230)',
               'rgb(0,0,204)','rgb(0,0,179)','rgb(0,0,153)',
               'rgb(0,0,128)','rgb(0,0,102)','rgb(0,0,77)',
               'rgb(0,0,51)','rgb(0,0,26)','rgb(0,0,0)']
color_brewer = list(reversed(color_brewer))
shapes = []
annotations = []
counter = 0

'''
The following loop creates subdicts taking the above information.
1 dictionary each is made for each treemap rectangle specifying 
additional properties like line, color and type = 'rect'
'''
for r in rects:
    shapes.append( 
        dict(
            type = 'rect', 
            x0 = r['x'], 
            y0 = r['y'], 
            x1 = r['x']+r['dx'], 
            y1 = r['y']+r['dy'],
            line = dict( width = 2 ),
            fillcolor = color_brewer[counter]
        ) 
    )
    annotations.append(
        dict(
            x = r['x']+(r['dx']/2),
            y = r['y']+(r['dy']/2),
            text = bowlers_1[counter], 
            showarrow = False
        )
    )
    counter = counter + 1
    if counter >= len(color_brewer):
        counter = 0

        
# For hover text
trace0 = go.Scatter(
    x = [ r['x']+(r['dx']/2) for r in rects ], 
    y = [ r['y']+(r['dy']/2) for r in rects ],
    text = values,
    mode = 'text',
)
        
layout = dict(
    height=700, 
    width=700,
    xaxis=dict(showgrid=False,zeroline=False),
    yaxis=dict(showgrid=False,zeroline=False),
    shapes=shapes,
    annotations=annotations,
    hovermode='closest'
)

# With hovertext
figure = dict(data=[trace0], layout=layout)

# Without hovertext
# figure = dict(data=[Scatter()], layout=layout)

iplot(figure, filename='squarify-treemap')