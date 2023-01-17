import pandas as pd
import numpy as np
import plotly.offline as plt
import plotly.graph_objs as go
import plotly
plotly.offline.init_notebook_mode(connected=True)
iris = pd.read_csv('../input/iris.csv')
iris.head()
Mapping = {'species' : 'class_name'}
iris.rename(index=str, columns=Mapping, inplace=True)
value = { name: i for i,name in enumerate(iris.class_name.unique())}
class_name = iris.class_name.copy()
iris.class_name = iris.class_name.map(value)
color = iris.class_name.copy()
color = color * 50
trace = go.Scatter(x=iris.sepal_length, y=iris.sepal_width, mode='markers', text=class_name,
                  marker=dict(size=16, color = color, colorscale='Viridis', showscale=True))

layout= go.Layout(
    title= 'Sepal Distribution',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Sepal Length',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title='Sepal Width',
        ticklen=5,
        gridwidth=2,
    ),
    showlegend=False
)

figure = go.Figure(data=[trace], layout=layout)

plt.iplot(figure)
trace = go.Scatter(x=iris.petal_length, y=iris.petal_width, mode='markers', text=class_name,
                  marker=dict(size=16, color = color, colorscale='Viridis', showscale=True))

layout= go.Layout(
    title= 'Petal Distribution',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Petal Length',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title='Petal Width',
        ticklen=5,
        gridwidth=2,
    ),
    showlegend=False
)

figure = go.Figure(data=[trace], layout=layout)

plt.iplot(figure)
areaSepal = iris.sepal_length * iris.sepal_width
areaPetal = iris.petal_length * iris.petal_width
trace = go.Scatter(y=areaSepal, mode='markers', text=class_name,
                  marker=dict(size=16, color = color, colorscale='Viridis', showscale=True))

layout= go.Layout(
    title= 'Sepal Area distribution',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Area',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title='Range',
        ticklen=5,
        gridwidth=2,
    ),
    showlegend=False
)

figure = go.Figure(data=[trace], layout=layout)

plt.iplot(figure)
trace = go.Scatter(y=areaPetal, mode='markers', text=class_name,
                  marker=dict(size=16, color = color, colorscale='Viridis', showscale=True))

layout= go.Layout(
    title= 'Petal Area distribution',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Area',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title='Range',
        ticklen=5,
        gridwidth=2,
    ),
    showlegend=False
)

figure = go.Figure(data=[trace], layout=layout)

plt.iplot(figure)
iris['sepal_area'] = iris.sepal_length * iris.sepal_width
iris['petal_area'] = iris.petal_length * iris.petal_width
class_name = iris.class_name.copy()
iris.drop('class_name', axis=1, inplace=True)
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris, class_name, test_size=0.2, random_state=78)
cluster = KMeans(n_clusters=3, n_jobs=-1, random_state=47, verbose=True, max_iter=300, n_init=2)
cluster.fit(x_train, y_train)
trace = go.Scatter(x=cluster.labels_, mode='markers', text=y_train,
                  marker=dict(size=16, color = y_train, colorscale='Viridis', showscale=True))

layout= go.Layout(
    title= 'cluster.labels_',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Class',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title='value',
        ticklen=5,
        gridwidth=2,
    ),
    showlegend=False
)

figure = go.Figure(data=[trace], layout=layout)

plt.iplot(figure)
y_pred = cluster.predict(x_test)
trace = go.Scatter(x=y_pred, mode='markers', text=y_test,
                  marker=dict(size=16, color = y_test, colorscale='Viridis', showscale=True))

layout= go.Layout(
    title= 'Prediction distribution',
    hovermode= 'closest',
    xaxis= dict(
        title= 'ClassType',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title='Value',
        ticklen=5,
        gridwidth=2,
    ),
    showlegend=False
)

figure = go.Figure(data=[trace], layout=layout)

plt.iplot(figure)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
print(cluster.score(x_test, y_test))
