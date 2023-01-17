# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn import datasets, linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import seaborn as sns             # görselleştirme yapmak için kullanacağız.
import missingno                  # eksik verileri daha iyi okumak için kullanacağız.
from sklearn import preprocessing   # ön işleme aşamasında label encoding vb. için dahil ettik.
import re    
df = pd.read_csv("../input/spotifyclassification/data.csv").copy()
df.head(10) 
df.shape #2017x17 lik bir veri tabanımız var öyleyse bu veri tabanımızı biraz inceleyelim
df.dtypes
df.info()
df.isnull().sum() # ohh miss gibi hiçbir nan veri yok itina ile yapılmış :D
df.describe().T #tüm sayısal verilerin istatistiksel verilerini görüyoruz
corr = df.corr()
corr
sns.heatmap(corr, xticklabels = corr.columns.values, yticklabels = corr.columns.values);
df.columns
df.columns = ['MuzisyenID','acousticness', 'danceability', 'duration_ms', 'energy',
       'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
       'speechiness', 'tempo', 'time_signature', 'valence', 'target',
       'song_title', 'artist']
df.head()
df.artist.nunique() # maşallah 1343 tane de sanatçı varmış :D
df.artist
df.sort_values('energy', axis = 0, ascending = False).head(10)[["energy","artist","song_title",]]
df.sort_values('energy', axis = 0, ascending = False).tail(10)[["energy","artist","song_title",]]
df[(df['artist'] == "Eminem")].sort_values('energy', axis = 0, ascending=False)
sns.scatterplot(x = "danceability", y = "tempo", data = df); # tempo ile danceability arasında böyle bir ilişki görüyoruz
chosen = ["energy", "liveness", "tempo", "valence", "loudness", "speechiness", "acousticness", "danceability", "instrumentalness"]
text1 = df["artist"] + " - " + df["song_title"]
text2 = text1.values

# X = data_frame.drop(droppable, axis=1).values
X = df[chosen].values
y = df["danceability"].values

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
