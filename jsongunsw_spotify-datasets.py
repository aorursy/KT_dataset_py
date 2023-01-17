# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn import datasets, linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

%matplotlib inline

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/featuresdf.csv")
data.head(100)
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
specific = ["energy", "liveness", "tempo", "valence", "loudness", "speechiness", "acousticness", "danceability", "instrumentalness"]
text1 = data["artists"] + " - " + data["name"]
text2 = text1.values


X = data[specific].values
y = data["danceability"].values

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
        size=12,
        color=y
    )
)

fig = go.Figure(data=[trace])
py.iplot(fig, filename="test-graph")
print("Mean value for danceability:", data['danceability'].mean())
sns.distplot(data['danceability'])
plt.show()
print("Mean value for energy:", data['energy'].mean())
sns.distplot(data['energy'])
plt.show()
print("Mean value for mode:", data['mode'].mean())
sns.distplot(data['mode'])
plt.show()
print("Mean value for speechiness:", data['speechiness'].mean())
sns.distplot(data['speechiness'])
plt.show()
print("Mean value for acousticness:", data['acousticness'].mean())
sns.distplot(data['acousticness'])
plt.show()
print("Mean value for instrumentalness:", data['instrumentalness'].mean())
sns.distplot(data['instrumentalness'])
plt.show()
print("Mean value for liveness:", data['liveness'].mean())
sns.distplot(data['liveness'])
plt.show()
print("Mean value for valence:", data['valence'].mean())
sns.distplot(data['valence'])
plt.show()
numeric = data.drop(['id','name','artists'], axis=1)
small = numeric.drop(['tempo','duration_ms','key','loudness','time_signature'], axis=1)
sns.set_palette('pastel')
small.mean().plot.bar()
plt.title('Mean Values of Audio Features')
plt.show()