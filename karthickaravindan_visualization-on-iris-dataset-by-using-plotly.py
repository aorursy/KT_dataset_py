import numpy as np 
import pandas as pd
import os
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
print(os.listdir("../input"))
from statistics import mean 

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.offline as py
from plotly import tools

py.init_notebook_mode(connected=True)
df = pd.read_csv("../input/Iris.csv")
table = ff.create_table(df)
py.iplot(table, filename='jupyter-table1')
set(df["Species"])
df.isnull().sum()
df['Species'].value_counts()
df_setosa=df[df['Species']=='Iris-setosa']
df_virginica=df[df['Species']=='Iris-virginica']
df_versicolor=df[df['Species']=='Iris-versicolor']

setosa = go.Scatter(x = df['SepalLengthCm'][df.Species =='Iris-setosa'], y = df['SepalWidthCm'][df.Species =='Iris-setosa']
                   , mode = 'markers', name = 'setosa')
versicolor = go.Scatter(x = df['SepalLengthCm'][df.Species =='Iris-versicolor'], y = df['SepalWidthCm'][df.Species =='Iris-versicolor']
                   , mode = 'markers', name = 'versicolor')
virginica = go.Scatter(x = df['SepalLengthCm'][df.Species =='Iris-virginica'], y = df['SepalWidthCm'][df.Species =='Iris-virginica']
                   , mode = 'markers', name = 'virginica')
data = [setosa, versicolor, virginica]
fig = dict(data=data)
py.iplot(fig, filename='styled-scatter')
setosa = go.Scatter(x = df['PetalLengthCm'][df.Species =='Iris-setosa'], y = df['PetalWidthCm'][df.Species =='Iris-setosa']
                   , mode = 'markers', name = 'setosa')
versicolor = go.Scatter(x = df['PetalLengthCm'][df.Species =='Iris-versicolor'], y = df['PetalWidthCm'][df.Species =='Iris-versicolor']
                   , mode = 'markers', name = 'versicolor')
virginica = go.Scatter(x = df['PetalLengthCm'][df.Species =='Iris-virginica'], y = df['PetalWidthCm'][df.Species =='Iris-virginica']
                   , mode = 'markers', name = 'virginica')
data = [setosa, versicolor, virginica]
fig = dict(data=data)
py.iplot(fig, filename='styled-scatter')
trace0 = go.Box(y=df['PetalWidthCm'][df['Species'] == 'Iris-setosa'],boxmean=True, name = 'setosa')

trace1 = go.Box(y=df['PetalWidthCm'][df['Species'] == 'Iris-versicolor'],boxmean=True, name = 'versicolor')

trace2 = go.Box(y=df['PetalWidthCm'][df['Species'] == 'Iris-virginica'],boxmean=True, name = 'virginica')

data = [trace0, trace1, trace2]
py.iplot(data)
trace1 = go.Scatter(
    y = df_setosa["SepalLengthCm"],
    mode='markers',
    marker=dict(
        size=16,
        color = 300, #set color equal to a variable
        colorscale='Viridis',
        showscale=True
    )
)

trace2 = go.Scatter(
    y = df_setosa["SepalWidthCm"],
    mode='markers',
    marker=dict(
        size=16,
        color = 200, #set color equal to a variable
        colorscale='Viridis',
        showscale=True
    )
)


data = [trace1,trace2]

py.iplot(data, filename='scatter-plot-with-colorscale')

classes=np.unique(df['Species'].values).tolist()
class_code={classes[k]: k for k in range(3)}
color_vals=[class_code[cl] for cl in df['Species']]
pl_colorscale=[[0.0, '#19d3f3'],
               [0.333, '#19d3f3'],
               [0.333, '#e763fa'],
               [0.666, '#e763fa'],
               [0.666, '#636efa'],
               [1, '#636efa']]

text=[df.loc[ k, 'Species'] for k in range(len(df))]

trace1 = go.Splom(dimensions=[dict(label='sepal length',
                                 values=df['SepalLengthCm']),
                            dict(label='sepal width',
                                 values=df['SepalWidthCm']),
                            dict(label='petal length',
                                 values=df['PetalLengthCm']),
                            dict(label='petal width',
                                 values=df['PetalWidthCm'])],
                text=text,
                #default axes name assignment :
                #xaxes= ['x1','x2',  'x3'],
                #yaxes=  ['y1', 'y2', 'y3'], 
                marker=dict(color=color_vals,
                            size=7,
                            colorscale=pl_colorscale,
                            showscale=False,
                            line=dict(width=0.5,
                                      color='rgb(230,230,230)'))
                )

axis = dict(showline=True,
          zeroline=False,
          gridcolor='#fff',
          ticklen=4)

layout = go.Layout(
    title='Iris Data set',
    dragmode='select',
    width=600,
    height=600,
    autosize=False,
    hovermode='closest',
    plot_bgcolor='rgba(240,240,240, 0.95)',
    xaxis1=dict(axis),
    xaxis2=dict(axis),
    xaxis3=dict(axis),
    xaxis4=dict(axis),
    yaxis1=dict(axis),
    yaxis2=dict(axis),
    yaxis3=dict(axis),
    yaxis4=dict(axis)
)

fig1 = dict(data=[trace1], layout=layout)
py.iplot(fig1, filename='splom-iris1')
