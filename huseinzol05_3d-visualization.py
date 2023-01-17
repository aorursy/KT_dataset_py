import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.offline as py

py.init_notebook_mode(connected = True)

import plotly.graph_objs as go

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Normalizer

from sklearn.decomposition import PCA
df = pd.read_csv('../input/fashion-mnist_train.csv', header = None)



# defined labels

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',

         'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



# defined colors, i love this one

colors = ['rgb(0,31,63)', 'rgb(255,133,27)', 'rgb(255,65,54)', 'rgb(0,116,217)', 'rgb(133,20,75)', 'rgb(57,204,204)',

'rgb(240,18,190)', 'rgb(46,204,64)', 'rgb(1,255,112)', 'rgb(255,220,0)',

'rgb(76,114,176)', 'rgb(85,168,104)', 'rgb(129,114,178)', 'rgb(100,181,205)']

df.head()
from sklearn.cross_validation import train_test_split



# visualize about 500 data

_, df_copy = train_test_split(df, test_size = 0.05)

df_copy_label = df_copy.iloc[:, 0]

df_copy_images = df_copy.iloc[:, 1:]
df_copy_images_ = StandardScaler().fit_transform(df_copy_images)

# push the data to different boundary

df_copy_images_ = Normalizer().fit_transform(df_copy_images_)

df_copy_images_component = PCA(n_components = 2).fit_transform(df_copy_images_)



from ast import literal_eval



plt.rcParams["figure.figsize"] = [21, 18]

for k, i in enumerate(np.unique(df_copy_label)):

    plt.scatter(df_copy_images_component[df_copy_label == i, 0],

               df_copy_images_component[df_copy_label == i, 1],

               color = '#%02x%02x%02x' % literal_eval(colors[k][3:]), 

                label = labels[k])

plt.legend()

plt.show()
from sklearn.manifold import TSNE

df_copy_images_dist = TSNE(n_components = 2).fit_transform(df_copy_images)



from ast import literal_eval



plt.rcParams["figure.figsize"] = [21, 18]

for k, i in enumerate(np.unique(df_copy_label)):

    plt.scatter(df_copy_images_dist[df_copy_label == i, 0],

               df_copy_images_dist[df_copy_label == i, 1],

               color = '#%02x%02x%02x' % literal_eval(colors[k][3:]), 

                label = labels[k])

plt.legend()

plt.show()
# stick to low boundary

df_copy_images_ = StandardScaler().fit_transform(df_copy_images)

df_copy_images_dist = TSNE(n_components = 3).fit_transform(df_copy_images_)



data_graph = []

for no, _ in enumerate(np.unique(df_copy_label)):

    graph = go.Scatter3d(

    x = df_copy_images_dist[df_copy_label == no, 0],

    y = df_copy_images_dist[df_copy_label == no, 1],

    z = df_copy_images_dist[df_copy_label == no, 2],

    name = labels[no],

    mode = 'markers',

    marker = dict(

        size = 12,

        line = dict(

            color = '#%02x%02x%02x' % literal_eval(colors[no][3:]),

            width = 0.5

            ),

        opacity = 0.5

        )

    )

    data_graph.append(graph)

    

layout = go.Layout(

    scene = dict(

        camera = dict(

            eye = dict(

            x = 0.5,

            y = 0.5,

            z = 0.5

            )

        )

    ),

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)

fig = go.Figure(data = data_graph, layout = layout)

py.iplot(fig, filename = '3d-scatter')
df_copy_images_ = StandardScaler().fit_transform(df_copy_images)

# push the data to different boundary

df_copy_images_ = Normalizer().fit_transform(df_copy_images_)

df_copy_images_component = PCA(n_components = 3).fit_transform(df_copy_images_)



data_graph = []

for no, _ in enumerate(np.unique(df_copy_label)):

    graph = go.Scatter3d(

    x = df_copy_images_component[df_copy_label == no, 0],

    y = df_copy_images_component[df_copy_label == no, 1],

    z = df_copy_images_component[df_copy_label == no, 2],

    name = labels[no],

    mode = 'markers',

    marker = dict(

        size = 12,

        line = dict(

            color = '#%02x%02x%02x' % literal_eval(colors[no][3:]),

            width = 0.5

            ),

        opacity = 0.5

        )

    )

    data_graph.append(graph)

    

layout = go.Layout(

    scene = dict(

        camera = dict(

            eye = dict(

            x = 0.5,

            y = 0.5,

            z = 0.5

            )

        )

    ),

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)

fig = go.Figure(data = data_graph, layout = layout)

py.iplot(fig, filename = '3d-scatter')