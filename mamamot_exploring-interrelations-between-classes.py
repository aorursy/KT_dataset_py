import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

import plotly.figure_factory as ff

init_notebook_mode(connected=True)
train = pd.read_csv('../input/train.csv')

train.index = train['id']

x_train = train['comment_text']

y_train = train.iloc[:, 2:]
correlation_matrix = y_train.corr()
heatmap = go.Heatmap(

    z=np.flip(correlation_matrix.values, axis=1),  # try it without flipping - looks unusual

    x=y_train.columns[::-1],

    y=y_train.columns,

    showscale=False,

    colorscale="viridis"

)



layout = go.Layout(

    title="Correlation between target variables",

    showlegend=False,

    width=700, height=700,

    autosize=False,

    margin=go.Margin(l=100, r=100, b=100, t=100, pad=4)

)



fig = go.Figure(data=[heatmap], layout=layout)

iplot(fig, filename='hmap')
dim = y_train.shape[1]

cooccurence_matrix = np.zeros((dim, dim))



for i in range(dim):

    for j in range(dim):

        res = sum(y_train.iloc[:, i] & y_train.iloc[:, j]) / sum(y_train.iloc[:, i])

        cooccurence_matrix[i, j] = res
heatmap = go.Heatmap(

    z=np.flip(cooccurence_matrix, axis=1),

    x=y_train.columns[::-1],

    y=y_train.columns,

    showscale=False,

    colorscale="viridis"

)



layout = go.Layout(

    title="Coocurence of target variables",

    showlegend=False,

    width=700, height=700,

    autosize=False,

    margin=go.Margin(l=100, r=100, b=100, t=100),

    yaxis=dict(

        title='If...'

    ),

    xaxis=dict(

        title='then...'

    )

)



fig = go.Figure(data=[heatmap], layout=layout)

iplot(fig, filename='hmap')