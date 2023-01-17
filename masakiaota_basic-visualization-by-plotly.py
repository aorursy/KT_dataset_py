##Visualization
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()

##import
import pandas as pd
import numpy as np
##acquire data
df = pd.read_csv('../input/creditcard.csv')
df0 = df[df.Class == 0]
df1 = df[df.Class == 1]
df1.head()
##make trace
trace0 = go.Histogram(
    x = df0.Amount,
    opacity = 0.7,
    name = 'class0',
    xbins = dict(
        start = 0,
        end = max(df.Amount),
        size = 50
    )
)
trace1 = go.Histogram(
    x = df1.Amount,
    opacity = 0.7,
    name = 'class1',
    xbins = dict(
        start = 0,
        end = max(df.Amount),
        size = 50
    )
)
data = [trace0, trace1]

##define layout
layout = go.Layout(
    #barmode='overlay',
    yaxis=dict(
        type='log',
        autorange=True,
        title = 'frequency'
    ),
    xaxis=dict(
        autorange=True,
        title = 'Amount'
    ),
    bargap=0.1,
    bargroupgap=0,
)

fig = go.Figure(data=data, layout=layout)

offline.iplot(fig)
##make trace
trace0 = go.Histogram(
    x = df0.V2,
    opacity = 0.7,
    name = 'class0',
    xbins = dict(
        start = min(df.V2),
        end = max(df.V2),
        size = 5
    )
)
trace1 = go.Histogram(
    x = df1.V2,
    opacity = 0.7,
    name = 'class1',
    xbins = dict(
        start = min(df.V2),
        end = max(df.V2),
        size = 5
    )
)
data = [trace0, trace1]

##define layout
layout = go.Layout(
    #barmode='overlay',
    yaxis=dict(
        type='log',
        autorange=True,
        title = 'frequency'
    ),
    xaxis=dict(
        autorange=True,
        title = 'V2'
    ),
    bargap=0.1,
    bargroupgap=0.05,
)

fig = go.Figure(data=data, layout=layout)

offline.iplot(fig)
print('Class 0:',len(df0),', Class 1:',len(df1))
##random under sampling
df0u = df0.sample(frac = 0.05)
print('Class 0:',len(df0u),', Class 1:',len(df1))
## make trace
trace0 = go.Scatter3d(
    x = df0u.V1,
    y = df0u.V2,
    z = df0u.V3,
    name = 'class0',
    mode = 'markers',
    opacity = 0.4,
    marker = dict(
        size = 2
    )
)
trace1 = go.Scatter3d(
    x = df1.V1,
    y = df1.V2,
    z = df1.V3,
    name = 'class1',
    mode = 'markers',
    marker = dict(
        size = 3
    )
)
## concatnate traces
data = [trace0, trace1]

## define layout
layout = go.Layout(
    title='3D-PCA',
    width=600,
    height=500,
    scene = dict(
        xaxis = dict(
            nticks=4, range = [min(df.V1),max(df.V1)], title='V1'),
        yaxis = dict(
            nticks=4, range = [min(df.V2),max(df.V2)], title='V2'),
        zaxis = dict(
            nticks=4, range = [min(df.V3),max(df.V3)], title='V3')
    ),
    showlegend=True)

fig = dict(data=data, layout=layout)
offline.iplot(fig)
