# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Good for interactive plots

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

from plotly import figure_factory as FF

init_notebook_mode()







# Add table data

table_data = [['Team', 'Wins', 'Losses', 'Ties'],

              ['Montréal<br>Canadiens', 18, 4, 0],

              ['Dallas Stars', 18, 5, 0],

              ['NY Rangers', 16, 5, 0], 

              ['Boston<br>Bruins', 13, 8, 0],

              ['Chicago<br>Blackhawks', 13, 8, 0],

              ['LA Kings', 13, 8, 0],

              ['Ottawa<br>Senators', 12, 5, 0]]

# Initialize a figure with FF.create_table(table_data)

#figure = py.figure_factory.create_table(table_data, height_constant=60)

figure = FF.create_table(table_data, height_constant=60)



# Add graph data

teams = ['Montréal Canadiens', 'Dallas Stars', 'NY Rangers',

         'Boston Bruins', 'Chicago Blackhawks', 'LA Kings', 'Ottawa Senators']

GFPG = [3.54, 3.48, 3.0, 3.27, 2.83, 2.45, 3.18]

GAPG = [2.17, 2.57, 2.0, 2.91, 2.57, 2.14, 2.77]

# Make traces for graph

trace1 = go.Scatter(x=teams, y=GFPG,

                    marker=dict(color='#0099ff'),

                    name='Goals For<br>Per Game',

                    xaxis='x2', yaxis='y2')

trace2 = go.Scatter(x=teams, y=GAPG,

                    marker=dict(color='#404040'),

                    name='Goals Against<br>Per Game',

                    xaxis='x2', yaxis='y2')



# Add trace data to figure

figure['data'].extend(go.Data([trace1, trace2]))



# Edit layout for subplots

figure.layout.xaxis.update({'domain': [0, .5]})

figure.layout.xaxis2.update({'domain': [0.6, 1.]})

# The graph's yaxis MUST BE anchored to the graph's xaxis

figure.layout.yaxis2.update({'anchor': 'x2'})

figure.layout.yaxis2.update({'title': 'Goals'})

# Update the margins to add a title and see graph x-labels. 

figure.layout.margin.update({'t':50, 'b':100})

figure.layout.update({'title': '2016 Hockey Stats'})



# Plot!

#py.iplot(figure, filename='subplot_table')

iplot(figure)