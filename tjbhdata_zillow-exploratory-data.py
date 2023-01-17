import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
%matplotlib inline
import gc
# Any results you write to the current directory are saved as output.
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go

init_notebook_mode(connected=True) #do not miss this line

train = pd.read_csv('../input/train_2016_v2.csv')
passenger = pd.read_csv('../input/properties_2016.csv',\
                        dtype={'parcelid':'int32','taxamount':'float32'})
print(train.shape)
print(passenger.shape)
traindf = pd.merge(train,passenger,how = 'left', on = 'parcelid')
traindf=traindf[ traindf.logerror > -0.4 ]
traindf=traindf[ traindf.logerror < 0.419 ]
traindf.head(5)
data = [go.Histogram(x=traindf['logerror'])]


layout = go.Layout(
    title='Histogram distribution of errors',
    xaxis=dict(
        title='Errors'
    ),
    yaxis=dict(
        title='Counts'
    ),
    bargap=0.2,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='error distribution histogram')
traindf['decade'] = (traindf['yearbuilt']//10)*10
x_data = ['decade1900','decade1910','decade1920','decade1930','decade1940','decade1950','decade1960', 'decade1970',
          'decade1980', 'decade1990',
          'decade2000', 'decade2010']

y0 = traindf.loc[traindf['decade'] == 1900,'logerror']
y1 = traindf.loc[traindf['decade'] == 1910,'logerror']
y2 = traindf.loc[traindf['decade'] == 1920,'logerror']
y3 = traindf.loc[traindf['decade'] == 1930,'logerror']
y4 = traindf.loc[traindf['decade'] == 1940,'logerror']
y5 = traindf.loc[traindf['decade'] == 1950,'logerror']
y6 = traindf.loc[traindf['decade'] == 1960,'logerror']
y7 = traindf.loc[traindf['decade'] == 1970,'logerror']
y8 = traindf.loc[traindf['decade'] == 1980,'logerror']
y9 = traindf.loc[traindf['decade'] == 1990,'logerror']
y10 = traindf.loc[traindf['decade'] == 2000,'logerror']
y11 = traindf.loc[traindf['decade'] == 2010,'logerror']


y_data = [y0,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11]

colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)',
         'rgba(93, 164, 214, 0.5)','rgba(255, 144, 14, 0.5)','rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)',
         'rgba(93, 164, 214, 0.5)']

traces = []

for xd, yd, cls in zip(x_data, y_data, colors):
        traces.append(go.Box(
            y=yd,
            name=xd,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=cls,
            marker=dict(
                size=2,
            ),
            line=dict(width=1),
        ))

layout = go.Layout(
    title='Error distribution for houses built over time (in decade)',
    yaxis=dict(
        autorange=False,
        showgrid=True,
        zeroline=True,
        dtick=5,
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2,
                range=[-2, 2],
        title='Errors'

    ),
    margin=dict(
        l=40,
        r=30,
        b=80,
        t=100,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    showlegend=False
)

fig = go.Figure(data=traces, layout=layout)
py.offline.iplot(fig, filename='error over decades')
traindf_gby = traindf.groupby(['yearbuilt'])['logerror'].agg(['median'])
traindf_gby.reset_index(drop = False,inplace = True)
plt.figure(figsize = (15,10))
plt.scatter(traindf_gby.loc[abs(traindf_gby['median']<= 0.3),'yearbuilt'],traindf_gby.loc[abs(traindf_gby['median']<= 0.3),'median'])
plt.style.use('ggplot')
plt.title('Error distribution in last century')
plt.ylabel('Error')
plt.xlabel('Year')
plt.ylim([-0.05,0.05])
plt.show()
N = 3000
dfsmall = traindf.sample(N, replace=True)


layout3 = go.Layout(
    title='Expansion of Los Angeles',
    
    yaxis=dict(
        title='Latitude'
    ),
    
    xaxis=dict(
        title='Longitude'
    ),
    
    )

trace = go.Scattergl(
    #title='Expansion of the City of Los Angeles',
    x = dfsmall['longitude'],
    y = dfsmall['latitude'],
    mode = 'markers',
    marker = dict(
        color = dfsmall['decade'],
        colorscale='Jet',
        line = dict(width = 1),
                showscale=True

    )
)
data3 = [trace]
fig3 = go.Figure(data=data3, layout=layout3)
py.offline.iplot(fig3, filename='error over decades')
x_data = ['decade1900','decade1910','decade1920','decade1930','decade1940','decade1950',
          'decade1960', 
          'decade1970',
          'decade1980', 'decade1990',
          'decade2000', 'decade2010'
         ]

y0 = dfsmall.loc[traindf['decade'] == 1900,'calculatedfinishedsquarefeet']
y1 = dfsmall.loc[traindf['decade'] == 1910,'calculatedfinishedsquarefeet']
y2 = dfsmall.loc[traindf['decade'] == 1920,'calculatedfinishedsquarefeet']
y3 = dfsmall.loc[traindf['decade'] == 1930,'calculatedfinishedsquarefeet']
y4 = dfsmall.loc[traindf['decade'] == 1940,'calculatedfinishedsquarefeet']
y5 = dfsmall.loc[traindf['decade'] == 1950,'calculatedfinishedsquarefeet']
y6 = dfsmall.loc[traindf['decade'] == 1960,'calculatedfinishedsquarefeet']
y7 = dfsmall.loc[traindf['decade'] == 1970,'calculatedfinishedsquarefeet']
y8 = dfsmall.loc[traindf['decade'] == 1980,'calculatedfinishedsquarefeet']
y9 = dfsmall.loc[traindf['decade'] == 1990,'calculatedfinishedsquarefeet']
y10 = dfsmall.loc[traindf['decade'] == 2000,'calculatedfinishedsquarefeet']
y11 = traindf.loc[traindf['decade'] == 2010,'calculatedfinishedsquarefeet']


y_data = [y0,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11]

colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)',
        'rgba(93, 164, 214, 0.5)','rgba(255, 144, 14, 0.5)','rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)',
         'rgba(93, 164, 214, 0.5)'
         ]

traces = []

for xd, yd, cls in zip(x_data, y_data, colors):
        traces.append(go.Box(
            y=yd,
            name=xd,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=cls,
            marker=dict(
                size=2,
            ),
            line=dict(width=1),
        ))

layout = go.Layout(
    title='Size of houses built over time (in decade)',
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        dtick=1000,
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2,
                #range=[-2, 2],
        title='area in SQFt '

    ),
    margin=dict(
        l=40,
        r=30,
        b=80,
        t=100,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    showlegend=False
)

fig = go.Figure(data=traces, layout=layout)
py.offline.iplot(fig, filename='error over decades')

