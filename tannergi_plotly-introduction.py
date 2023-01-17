import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
iris = pd.read_csv('../input/iris/Iris.csv')
iris.drop(['Id'], axis=1, inplace=True)
iris.rename(columns={'SepalLengthCm': 'sepal_length', 'SepalWidthCm': 'sepal_width', 'PetalLengthCm': 'petal_length', 'PetalWidthCm': 'petal_width', 'Species': 'label'}, inplace=True)
iris.head()
wine_reviews = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv', index_col=0)
wine_reviews.head()
trace1 = go.Scatter(
    y = iris.sepal_length,
    mode = 'lines+markers',
    line = dict(
        color = 'rgb(255, 53, 53)',
        width=3
    )
)

layout = go.Layout(
    title = 'Sepal Length',
)

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
iplot({
    'data':[{'y': iris.sepal_length, 'mode': 'lines', 'line':dict(color = 'rgb(52, 127, 255)', width=3)}],
    'layout': {'title': 'Sepal Length'}
})
trace1 = go.Scatter(
    y = iris.sepal_length,
    mode = 'lines',
    name='Sepal Length'
)

trace2 = go.Scatter(
    y = iris.sepal_width,
    mode = 'lines',
    name='Sepal Width'
)

layout = go.Layout(
    title = 'Sepal Length and Width',
)

fig = go.Figure(data=[trace1, trace2], layout=layout)

iplot(fig)
columns = iris.drop(['label'], axis=1).columns

iplot({
    'data': [{'y': iris[col], 'name':col} for col in columns],
    'layout': {'title': 'Iris dataset'}
})
trace1 = go.Scatter(
    x = iris.sepal_width,
    y = iris.sepal_length,
    mode = 'markers',
)

layout = go.Layout(
    title = 'Sepal Length and Width',
    xaxis = dict(
        title = 'Sepal Width'
    ),
    yaxis = dict(
        title = 'Sepal Length'
    )
)

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
trace1 = go.Scatter(
    x = wine_reviews.head(1000)['points'], 
    y = wine_reviews.head(1000)['price'], 
    mode = 'markers'
)

layout = go.Layout(
    title = 'Wine Prices',
    xaxis = dict(
        title = 'Points'
    ),
    yaxis = dict(
        title = 'Price'
    )
)

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
trace1 = go.Scatter(
    x = wine_reviews.head(1000)['points'], 
    y = wine_reviews.head(1000)['price'], 
    mode = 'markers',
    marker = dict(
        size=10,
        color = 'rgba(152, 0, 0, 0.8)',
        line = dict(
            width=1
        )
    )
)

layout = go.Layout(
    title = 'Wine Prices',
    xaxis = dict(
        title = 'Points'
    ),
    yaxis = dict(
        title = 'Price'
    )
)

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
trace1 = go.Scatter(
    x = wine_reviews.head(1000)['points'], 
    y = wine_reviews.head(1000)['price'], 
    mode = 'markers',
    marker = dict(
        size=10,
        color = 'rgba(152, 0, 0, 0.8)',
        line = dict(
            width=1
        )
    ),
    text = wine_reviews.title
)

layout = go.Layout(
    title = 'Wine Prices',
    xaxis = dict(
        title = 'Points'
    ),
    yaxis = dict(
        title = 'Price'
    )
)

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
data = wine_reviews['points'].value_counts()

trace1 = go.Bar(
    x = data.index,
    y = data.values,
    marker = dict(
        color = 'rgb(237, 85, 103)',
        line = dict(
            color='rgb(0, 0, 0)',
            width=2
        )
    )
)

layout = go.Layout(
    title = 'Wine Points Frequency',
    xaxis = dict(
        title = 'Points'
    ),
    yaxis = dict(
        title = 'Frequency'
    )
)

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
data = wine_reviews['country'].value_counts()[:5]

trace1 = go.Bar(
    x = data.index,
    y = data.values,
    marker = dict(
        color = 'rgb(84, 237, 181)',
        line = dict(
            color='rgb(0, 0, 0)',
            width=2
        )
    )
)

layout = go.Layout(
    title = 'Top 5 producing Countries',
    xaxis = dict(
        title = 'Country'
    ),
    yaxis = dict(
        title = 'Frequency'
    )
)

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
data = iris.groupby('label').mean()
data.head()
traces = [go.Bar(
    x = data.index,
    y = data[col],
    name = col,
    marker = dict(
        line = dict(
            color = 'rgb(0, 0, 0)',
            width = 2
        )
    )
) for col in data.columns]

layout = go.Layout(
    title = 'Iris Dataset',
    xaxis = dict(
        title = 'Species',
    ),
    yaxis = dict(
        title = 'Length/Width in Cm'
    )
)

fig = go.Figure(data=traces, layout=layout)

iplot(fig)
traces = [go.Bar(
    x = data[col],
    y = data.index,
    orientation = 'h',
    name = col,
    marker = dict(
        line = dict(
            color = 'rgb(0, 0, 0)',
            width = 2
        )
    )
) for col in data.columns]

layout = go.Layout(
    title = 'Iris Dataset',
    xaxis = dict(
        title = 'Length/Width in Cm'
    ),
    yaxis = dict(
        title = 'Species'
    )
)

fig = go.Figure(data=traces, layout=layout)

iplot(fig)
traces = [go.Bar(
    x = data.index,
    y = data[col],
    name = col,
    marker = dict(
        line = dict(
            color = 'rgb(0, 0, 0)',
            width = 2
        )
    )
) for col in data.columns]

layout = go.Layout(
    title = 'Iris Dataset',
    barmode = 'stack',
    xaxis = dict(
        title = 'Species',
    ),
    yaxis = dict(
        title = 'Length/Width in Cm'
    )
)

fig = go.Figure(data=traces, layout=layout)

iplot(fig)
data = wine_reviews['points']

trace1 = go.Histogram(
    x = data,
    marker = dict(
        color = 'rgb(51, 83, 214)',
        line = dict(
            color='rgb(0, 0, 0)',
            width=2
        )
    )
)

layout = go.Layout(
    title = 'Wine Points Frequency',
    xaxis = dict(
        title = 'Points'
    ),
    yaxis = dict(
        title = 'Frequency'
    )
)

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
data = wine_reviews['points']

trace1 = go.Histogram(
    x = data,
    xbins = dict(
        start = 80,
        end = 100,
        size=5
    ),
    marker = dict(
        color = 'rgb(51, 83, 214)',
        line = dict(
            color='rgb(0, 0, 0)',
            width=2
        )
    )
)

layout = go.Layout(
    title = 'Wine Points Frequency',
    xaxis = dict(
        title = 'Points'
    ),
    yaxis = dict(
        title = 'Frequency'
    )
)

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
data = wine_reviews['price']

trace1 = go.Histogram(
    x = data,
    xbins = dict(
        start = 0,
        end = 150,
        size=25
    ),
    marker = dict(
        color = 'rgb(237, 166, 14)',
        line = dict(
            color='rgb(0, 0, 0)',
            width=2
        )
    )
)

layout = go.Layout(
    title = 'Wine Price Frequencies',
    xaxis = dict(
        title = 'Price'
    ),
    yaxis = dict(
        title = 'Frequency'
    )
)

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
data = wine_reviews['country'].value_counts()[:5]

trace1 = go.Pie(
    values = data.values,
    labels = data.index
)

layout = go.Layout(
    title = 'Top 5 Wine Producing Countries'
)

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
data = wine_reviews['points'][(wine_reviews['points']>=90) & (wine_reviews['points']<=95)].value_counts()

colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1', '#fcbb2f'] 

trace1 = go.Pie(
    values = data.values,
    labels = data.index,
    hoverinfo = 'label+percent+value',
    textinfo = 'value',
    textfont = dict(
        size=20
    ),
    marker = dict(
        colors = colors,
        line = dict(
            color = 'rgb(0, 0, 0)',
            width=2
        )
    )
)

layout = go.Layout(
    title = 'Wine Point Frequency'
)

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
data = [go.Box(
    y = iris[col],
    name = col
) for col in iris.drop(['label'], axis=1).columns]

layout = go.Layout(
    title = 'Iris Dataset Distribution'
)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
data = [go.Box(
    x = iris[col],
    name = col
) for col in iris.drop(['label'], axis=1).columns]

layout = go.Layout(
    title = 'Iris Dataset Distribution'
)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
wines = wine_reviews['variety'].value_counts().index[:5]
data = [wine_reviews['price'][(wine_reviews['variety']==wine) & (wine_reviews['price']<300)] for wine in wines]

data = [go.Box(
    y = col,
    name = wines[i],
    boxpoints = 'outliers'
) for i, col in enumerate(data)]

layout = go.Layout(
    title = 'Wine Kinds'
)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
wines = wine_reviews['variety'].value_counts().index[:5]
data = [wine_reviews['price'][(wine_reviews['variety']==wine) & (wine_reviews['price']<300)] for wine in wines]

data = [go.Box(
    y = col,
    name = wines[i],
    boxpoints = False
) for i, col in enumerate(data)]

layout = go.Layout(
    title = 'Wine Kinds'
)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
data = [go.Violin(
    y = iris[col],
    name = col
) for col in iris.drop(['label'], axis=1).columns]

layout = go.Layout(
    title = 'Iris Dataset Distribution'
)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
data = [go.Violin(
    y = iris[col],
    name = col,
    box = dict(
        visible = True
    ),
    meanline = dict(
        visible = True
    )
) for col in iris.drop(['label'], axis=1).columns]

layout = go.Layout(
    title = 'Iris Dataset Distribution'
)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
x, y, z = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 400).transpose()

trace1 = go.Scatter3d(
    x = x,
    y = y,
    z = z,
    mode = 'markers',
    marker = dict(
        size = 12,
        color = z,
        colorscale = 'Viridis',
        opacity = 0.8
    )
)

layout = go.Layout(
    title = '3D Scatter Plot'
)

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
x, y, z = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 20).transpose()

trace1 = go.Scatter3d(
    x = x,
    y = y,
    z = z,
    mode = 'lines+markers',
    line = dict(
        width = 3,
    ),
    marker = dict(
        size = 6,
        color = z,
        colorscale = 'Viridis',
        opacity = 0.8
    )
)

layout = go.Layout(
    title = '3D Line Plot'
)

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
trace1 = go.Heatmap(
    z=np.array(iris.drop(['label'], axis=1).corr()),
    x = iris.drop(['label'], axis=1).columns,
    y = iris.drop(['label'], axis=1).columns,
    colorscale = 'Viridis'
)

layout = go.Layout(
    title = 'Heatmap'
)

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
trace1 = go.Scatter(
    y = iris.sepal_length,
    mode = 'lines',
    name = 'Sepal Length'
)

trace2 = go.Scatter(
    y = iris.sepal_width,
    mode = 'lines',
    name = 'Sepal Width',
    xaxis = 'x2',
    yaxis = 'y2'
)

trace3 = go.Scatter(
    y = iris.petal_length,
    mode = 'lines',
    name = 'Petal Length',
    xaxis = 'x3',
    yaxis = 'y3'
)

trace4 = go.Scatter(
    y = iris.petal_width,
    mode = 'lines',
    name = 'Petal Width',
    xaxis = 'x4',
    yaxis = 'y4'
)

data = [trace1, trace2, trace3, trace4]

layout = go.Layout(
    xaxis = dict(
        domain = [0, 0.45]
    ),
    yaxis = dict(
        domain = [0, 0.45]
    ),
    xaxis2 = dict(
        domain = [0.55, 1]
    ),
    xaxis3 = dict(
        domain = [0, 0.45],
        anchor = 'y3'
    ),
    xaxis4 = dict(
        domain = [0.55, 1],
        anchor = 'y4'
    ),
    yaxis2 = dict(
        domain = [0, 0.45],
        anchor = 'x2'
    ),
    yaxis3 = dict(
        domain = [0.55, 1]
    ),
    yaxis4 = dict(
        domain = [0.55, 1],
        anchor = 'x4'
    )
)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
from plotly import tools

trace1 = go.Scatter(
    y = iris.sepal_length,
    mode = 'lines',
    name = 'Sepal Length'
)

trace2 = go.Scatter(
    y = iris.sepal_width,
    mode = 'lines',
    name = 'Sepal Width',
)

trace3 = go.Scatter(
    y = iris.petal_length,
    mode = 'lines',
    name = 'Petal Length',
)

trace4 = go.Scatter(
    y = iris.petal_width,
    mode = 'lines',
    name = 'Petal Width',
)

fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'))

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)

fig['layout'].update(title='Iris Dataset')

iplot(fig)
trace1 = go.Splom(
    dimensions=[
        dict(
            label='sepal length',
            values=iris['sepal_length']
        ),
        dict(
            label='sepal width',
            values=iris['sepal_width']
        ),
        dict(
            label='petal length',
            values=iris['petal_length']
        ),
        dict(
            label='petal width',
            values=iris['petal_width']
        ) 
    ],
    text = iris.label
)

axis = dict(
    showline=True,
    zeroline=False,
    gridcolor='#fff',
    ticklen=4
)

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

fig = go.Figure(data=[trace1], layout=layout)
iplot(fig)
data = wine_reviews['country'].replace("US", "United States").value_counts()

trace1 = go.Choropleth(
    locationmode = 'country names',
    locations = data.index.values,
    text = data.index,
    z = data.values
)

layout = go.Layout(
    title = 'Wine Production Worldmap'
)

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)