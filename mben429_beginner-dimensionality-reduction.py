import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sb

import matplotlib

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

# load the Data

training = pd.read_csv("../input/train.csv")



target = training["label"]

training = training.drop("label", axis=1)



# set up figure

plt.figure(figsize=(15, 13))

for digit in range(0, 50):

    plt.subplot(5, 10, digit+1)

    data_grid = training.iloc[digit].values.reshape(28, 28)

    plt.imshow(data_grid)

plt.tight_layout(pad=1)

    

    
# Set up a variable N = number of rows to apply PCA to

N = 10000

X = training[:N].values



# Y = Target Values

Y = target[:N]



# Standardize the values

X_std = StandardScaler().fit_transform(X)



# Call PCA method from sklearn toolkit

pca = PCA(n_components=4)

principle_components = pca.fit_transform(X_std)
trace = go.Scatter(

    x = principle_components[:, 0],

    y = principle_components[:, 1],

    mode="markers",

    text=Y,

    marker=dict(

        size=8, color=Y, 

        colorscale="Electric", 

        opacity=0.7

    )

)



data = [trace]



layout = go.Layout(

    title="Principle Component Analysis",

    xaxis=dict(

        title="First Principle Component",

        gridwidth=2

    ),

    yaxis=dict(

        title="Second Princple Component",

        gridwidth=2

    )

)



fig = dict(data=data, layout=layout)

py.iplot(fig)
# Use Kmeans method from sklearn.cluster library

kmn = KMeans(init='k-means++', n_clusters=9, random_state=0)

X_kmean = kmn.fit_predict(principle_components)



trace_k = go.Scatter(

    x=principle_components[:,0],

    y=principle_components[:,1],

    mode="markers",

    marker=dict(

        size=8,

        color=X_kmean,

        colorscale="Picnic",

    )

)

data_k = [trace_k]



layout_k = go.Layout(

    title="K-Means Clustering result",

    xaxis=dict(title='First Principle Component', gridwidth=2),

    yaxis=dict(title='Second Principle Component', gridwidth=2)

)



fig_k = dict(data=data_k, layout=layout_k)

py.iplot(fig_k)


