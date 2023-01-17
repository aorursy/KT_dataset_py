import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

from sklearn import datasets, linear_model

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

from sklearn.manifold import TSNE



%matplotlib inline



data_frame = pd.read_csv("../input/data.csv")

data_frame = data_frame.drop("Unnamed: 0", axis="columns")

data_frame.head()
x = data_frame["danceability"].values

y = data_frame["valence"].values



x = x.reshape(x.shape[0], 1)

y = y.reshape(y.shape[0], 1)



regr = linear_model.LinearRegression()

regr.fit(x, y)



fig = plt.figure(figsize=(6, 6))

fig.suptitle("Correlation between danceability and song mood")



ax = plt.subplot(1, 1, 1)

ax.scatter(x, y, alpha=0.5)

ax.plot(x, regr.predict(x), color="red", linewidth=3)

plt.xticks(())

plt.yticks(())



ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))



ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))



plt.xlabel("danceability")

plt.ylabel("valence")



plt.show()
x = "danceability"

y = "valence"



fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, sharex=False, figsize=(10, 5))

fig.suptitle("Histograms")

h = ax2.hist2d(data_frame[x], data_frame[y], bins=20)

ax1.hist(data_frame["energy"])



ax2.set_xlabel(x)

ax2.set_ylabel(y)



ax1.set_xlabel("energy")



plt.colorbar(h[3], ax=ax2)



plt.show()
chosen = ["energy", "liveness", "tempo", "valence", "loudness", "speechiness", "acousticness", "danceability", "instrumentalness"]

text1 = data_frame["artist"] + " - " + data_frame["song_title"]

text2 = text1.values



# X = data_frame.drop(droppable, axis=1).values

X = data_frame[chosen].values

y = data_frame["danceability"].values



min_max_scaler = MinMaxScaler()

X = min_max_scaler.fit_transform(X)



pca = PCA(n_components=3)

pca.fit(X)



X = pca.transform(X)



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



trace = go.Scatter3d(

    x=X[:,0],

    y=X[:,1],

    z=X[:,2],

    text=text2,

    mode="markers",

    marker=dict(

        size=8,

        color=y

    )

)



fig = go.Figure(data=[trace])

py.iplot(fig, filename="test-graph")
chosen = ["energy", "liveness", "tempo", "valence"]

text1 = data_frame["artist"] + " - " + data_frame["song_title"]

text2 = text1.values



# X = data_frame.drop(droppable, axis=1).values

X = data_frame[chosen].values

y = data_frame["loudness"].values



min_max_scaler = MinMaxScaler()

X = min_max_scaler.fit_transform(X)



pca = PCA(n_components=2)

pca.fit(X)



X = pca.transform(X)



fig = {

    "data": [

        {

            "x": X[:, 0],

            "y": X[:, 1],

            "text": text2,

            "mode": "markers",

            "marker": {"size": "8", "color": y}

        }

    ],

    "layout": {

        "xaxis": {"title": "How hard is this to dance to?"},

        "yaxis": {"title": "How metal is this?"}

    }

}



py.iplot(fig, filename="test-graph2")
import time



chosen = ["energy", "liveness", "tempo", "valence", "loudness",

          "speechiness", "acousticness", "danceability", "instrumentalness"]



X = data_frame[chosen].values

y = data_frame["loudness"].values



min_max_scaler = MinMaxScaler()

X = min_max_scaler.fit_transform(X)



time_start = time.time()

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsne_results = tsne.fit_transform(X)



print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))



fig = {

    "data": [

        {

            "x": tsne_results[:, 0],

            "y": tsne_results[:, 1],

            "text": text2,

            "mode": "markers",

            "marker": {"size": "8", "color": y}

        }

    ],

    "layout": {

        "xaxis": {"title": "x-tsne"},

        "yaxis": {"title": "y-tsne"}

    }

}



py.iplot(fig, filename="test-graph2")