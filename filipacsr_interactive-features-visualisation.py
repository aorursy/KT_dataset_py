from __future__ import division

import pandas

import csv

import numpy as np

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.tools import FigureFactory as FF

from plotly.graph_objs import *

import plotly.tools as tls

import plotly

py.sign_in('xxx', 'xxx')
train = pandas.DataFrame.from_csv('../input/train.csv', index_col=None)

train[:5]
train = train.drop(['Id'], axis = 1)
y = train['SalePrice']

#logarithm of the salesprice 

y_log = np.log(y)

train = train.drop(['SalePrice'], axis = 1)
train[:5]
k = 0

fig_all = []

for ll in range(0, int(len(train.columns)/12 + 1)):

    

    j = i = 1



    fig = plotly.tools.make_subplots(rows=3, cols=4,subplot_titles=(train.columns[k:k + 12:]), shared_yaxes=False)

    

    ll = 1

    

    for feature in train.columns[k:k + 12:]:

        

        k = k + 1

    

        trace = go.Scatter(

            x = train[feature],

            y = y_log,

            mode='markers',

            marker=dict(

            size='5',

            color = y_log, 

            colorscale='magma',

            showscale=True, 

        ))

    

        fig.append_trace(trace, i, j)

        if j%4 == 0:

            i = i + 1

            j = 1

        else:

            j = j + 1

    

        fig['layout']['yaxis' + str(ll)].update(showgrid=False)

        fig['layout']['xaxis' + str(ll)].update(showgrid=False, tickangle = '-40')

        

        ll = ll + 1

        fig['layout'].update(showlegend=False)

 

    fig_all.append(fig)
#py.plot(fig_all[0], filename = 'plot1')

tls.embed("https://plot.ly/~fr/1893/")
#py.plot(fig_all[1], filename = 'plot2')

tls.embed("https://plot.ly/~fr/1895/")
#py.plot(fig_all[2], filename = 'plot3')

tls.embed("https://plot.ly/~fr/1897/")
#py.plot(fig_all[3], filename = 'plot4')

tls.embed("https://plot.ly/~fr/1899/")
#py.plot(fig_all[4], filename = 'plot5')

tls.embed("https://plot.ly/~fr/1901/")
#py.plot(fig_all[5], filename = 'plot6')

tls.embed("https://plot.ly/~fr/1903/")
#py.plot(fig_all[6], filename = 'plot7')

tls.embed("https://plot.ly/~fr/1905/")