import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# matplotlib configure

plt.rcParams['image.cmap'] = 'gray'

# Color from R ggplot colormap

color = ['#6388b4', '#ffae34', '#ef6f6a', '#8cc2ca', '#55ad89', '#c3bc3f', '#bb7693', '#baa094', '#a9b5ae', '#767676']
mnist = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

mnist.head()
label = mnist['label']

mnist.drop(['label'], inplace=True, axis=1)
def arr2img(arr, img_size=(28, 28)):

    return arr.reshape(img_size)



fig, axes = plt.subplots(2, 5, figsize=(10, 2))



for idx, ax in enumerate(axes.flat):

    ax.imshow(arr2img(mnist[idx:idx+1].values))

    ax.set_title(label[idx], fontweight='bold', fontsize=8)

    ax.axis('off')



plt.subplots_adjust(bottom=0.1, right=0.5, top=0.9)

plt.show()

from sklearn.decomposition import PCA



pca = PCA(n_components=2)

pca.fit(mnist)

mnist_pca = pca.transform(mnist)
import plotly.graph_objects as go



fig = go.Figure()



for idx in range(10):

    fig.add_trace(go.Scatter(

        x = mnist_pca[:,0][label==idx],

        y = mnist_pca[:,1][label==idx],

        name=str(idx),

        opacity=0.6,

        mode='markers',

        marker=dict(color=color[idx])

        

    ))



fig.update_layout(

    width = 800,

    height = 800,

    title = "PCA result",

    yaxis = dict(

      scaleanchor = "x",

      scaleratio = 1

    ),

    legend=dict(

        orientation="h",

        yanchor="bottom",

        y=1.02,

        xanchor="right",

        x=1

    )

)





fig.show()