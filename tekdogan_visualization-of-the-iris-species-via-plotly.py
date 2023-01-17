import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
data = pd.read_csv("../input/Iris.csv")
data.info()
data.head()
data_sorted_bySW = data.sort_values('SepalWidthCm')
data_sorted_byPL = data.sort_values('PetalLengthCm')
data_sorted_byPW = data.sort_values('PetalWidthCm')
df = data.iloc[:100, :]

bySW = go.Scatter(
                    x = data_sorted_bySW.SepalWidthCm,
                    y = data_sorted_bySW.SepalLengthCm,
                    mode = "markers",
                    name = "Sepal Width (cm)",
                    marker = dict(color = 'rgba(255, 0, 0, 0.9)'),
                    text = data_sorted_bySW.Species
)

byPL = go.Scatter(
                    x = data_sorted_byPL.PetalLengthCm,
                    y = data_sorted_byPL.SepalLengthCm,
                    mode = "markers",
                    name = "Petal Length (cm)",
                    marker = dict(color = 'rgba(0, 255, 0, 0.9)'),
                    text = data_sorted_byPL.Species
)

byPW = go.Scatter(
                    x = data_sorted_byPW.PetalWidthCm,
                    y = data_sorted_byPW.PetalWidthCm,
                    mode = "markers",
                    name = "Petal Width (cm)",
                    marker = dict(color = 'rgba(0, 0, 255, 0.9)'),
                    text = data_sorted_byPW.Species
)

layout = dict(title = 'Change of Sepal Length by Other Properties',
              xaxis= dict(title= 'centimeters',ticklen= 5,zeroline= False)
             )
u = [bySW, byPL, byPW]
fig = dict(data = u)
iplot(fig)

data1 = data.groupby(data.Species).mean()
data1['Species'] = data1.index

t1 = go.Bar(
            x = data1.Species,
            y = data1.SepalLengthCm,
            name = "Sepal Length (cm)",
            marker = dict(color = 'rgba(160, 55, 0, 0.8)', line = dict(color = 'rgb(0,0,0)', width = 1.5)),
            text = data1.Species
)

t2 = go.Bar(
            x = data1.Species,
            y = data1.SepalWidthCm,
            name = "Sepal Width (cm)",
            marker = dict(color = 'rgba(0, 55, 160, 0.8)', line = dict(color = 'rgb(0,0,0)', width = 1.5)),
            text = data1.Species
)

t3 = go.Bar(
            x = data1.Species,
            y = data1.PetalLengthCm,
            name = "Petal Length (cm)",
            marker = dict(color = 'rgba(20, 55, 30, 0.8)', line = dict(color = 'rgb(0,0,0)', width = 1.5)),
            text = data1.Species
)

t4 = go.Bar(
            x = data1.Species,
            y = data1.PetalWidthCm,
            name = "Petal Width (cm)",
            marker = dict(color = 'rgba(70, 55, 160, 0.8)', line = dict(color = 'rgb(0,0,0)', width = 1.5)),
            text = data1.Species
)

b = [t1,t2,t3,t4]
layout_bar = go.Layout(barmode = "group")
fig_bar = go.Figure(data = b, layout = layout_bar)
iplot(fig_bar)
fig_bubble = [
    {
        'x' : data.PetalLengthCm,
        'y' : data.PetalWidthCm,
        'mode' : 'markers',
        'marker' : {
            'color' : data.SepalWidthCm,
            'size' : data.SepalLengthCm,
            'showscale' : True
        },
        'text' : data.Species
    }
]
iplot(fig_bubble)
t1_box = go.Box(
                name = 'Sepal Length (cm)',
                y = data.SepalLengthCm,
                marker = dict(color = 'rgba(160,160,50,0.7)')
)

t2_box = go.Box(
                name = 'Sepal Width (cm)',
                y = data.PetalWidthCm,
                marker = dict(color = 'rgba(50,160,150,0.7)')
)

t3_box = go.Box(
                name = 'Petal Length (cm)',
                y = data.PetalLengthCm,
                marker = dict(color = 'rgba(160,60,150,0.7)')
)

t4_box = go.Box(
                name = 'Petal Width (cm)',
                y = data.SepalWidthCm,
                marker = dict(color = 'rgba(150,160,150,0.7)')
)

fig_box = [t1_box, t2_box, t3_box, t4_box]

iplot(fig_box)
import plotly.figure_factory as ff
data_ff = data.loc[:, ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
data_ff['index'] = np.arange(1, len(data_ff)+1)

fig_ff = ff.create_scatterplotmatrix(data_ff, diag = 'box', index = 'index', colormap = 'Blues', colormap_type = 'cat', height = 800, width = 800)
iplot(fig_ff)
trace_3d = go.Scatter3d(
                        x = data.SepalLengthCm,
                        y = data.SepalWidthCm,
                        z = data.PetalLengthCm,
                        mode = 'markers',
                        opacity = 0.7,
                        #name = data.Species,
                        marker = dict(
                                    size = 5,
                                    color = data.PetalWidthCm
                        )
)

list_3d = [trace_3d]

fig_3d = go.Figure(data = list_3d)
iplot(fig_3d)
i_setosa = data[data['Species']  == 'Iris-setosa']
i_versicolor = data[data['Species']  == 'Iris-versicolor']
i_virginica = data[data['Species']  == 'Iris-virginica']
# Iris-setosa
trace_setosa = go.Scatter3d(
                        x = i_setosa.SepalLengthCm,
                        y = i_setosa.SepalWidthCm,
                        z = i_setosa.PetalLengthCm,
                        mode = 'markers',
                        opacity = 0.7,
                        name = "Iris-setosa",
                        marker = dict(
                                    size = 5,
                                    color = 'rgba(255,102, 255,0.8)'
                        )
)

# Iris-versicolor
trace_versicolor = go.Scatter3d(
                        x = i_versicolor.SepalLengthCm,
                        y = i_versicolor.SepalWidthCm,
                        z = i_versicolor.PetalLengthCm,
                        mode = 'markers',
                        opacity = 0.7,
                        name = "Iris-versicolor",
                        marker = dict(
                                    size = 5,
                                    color = 'rgba(102, 255, 51, 0.8)'
                        )
)

# Iris-virginica
trace_virginica = go.Scatter3d(
                        x = i_virginica.SepalLengthCm,
                        y = i_virginica.SepalWidthCm,
                        z = i_virginica.PetalLengthCm,
                        mode = 'markers',
                        opacity = 0.7,
                        name = "Iris-virginica",
                        marker = dict(
                                    size = 5,
                                    color = 'rgba(51, 102, 255, 0.8)'
                        )
)

list_3d = [trace_setosa, trace_versicolor, trace_virginica]

fig_3d = go.Figure(data = list_3d)
iplot(fig_3d)
data.head()
