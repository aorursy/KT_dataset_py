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

label = mnist['label']

mnist.drop(['label'], inplace=True, axis=1)
%%time

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



lda = LDA(n_components=2)

lda.fit(mnist, label)

mnist_lda = lda.transform(mnist)
import plotly.graph_objects as go



fig = go.Figure()



for idx in range(10):

    fig.add_trace(go.Scatter(

        x = mnist_lda[:,0][label==idx],

        y = mnist_lda[:,1][label==idx],

        name=str(idx),

        opacity=0.6,

        mode='markers',

        marker=dict(color=color[idx])

        

    ))



fig.update_layout(

    width = 800,

    height = 800,

    title = "LDA result",

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