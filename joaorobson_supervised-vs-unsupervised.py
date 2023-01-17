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
df = pd.read_csv('../input/spotifyclassification/data.csv', index_col=0)
df

# Remover colunas com o nome da música e do artista, assim como a label de cada música

df.drop(columns=["target", "song_title", "artist"],  inplace=True)
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score, silhouette_samples



 



n_clusters_range = range(2, 15)

silhouetes_scrs = []

silhouetes_scrs_samples = []

for n_clusters in n_clusters_range:

    kmeans = KMeans(n_clusters=n_clusters).fit(df)

    silhouete_scr = silhouette_score(df, kmeans.labels_)

    silhouetes_scrs.append(silhouete_scr)

    silhouetes_scrs_samples.append(silhouette_samples(df, kmeans.labels_))

 

    
silhouetes_scrs
import matplotlib.pyplot as plt



plt.plot(n_clusters_range, silhouetes_scrs)

plt.xlabel("Número de clusters")

plt.ylabel("Coeficiente de silhueta")

plt.show()
kmeans = KMeans(n_clusters=2).fit(df)
from sklearn.manifold import TSNE



tsne = TSNE(n_components=2,  n_iter=1000, init='pca').fit_transform(df)

plt.figure(figsize=(10,10))

plt.scatter(tsne[:,0], tsne[:,1], c=kmeans.labels_);
pitchfork_df= pd.read_csv('../input/pitchfork-reviews-through-12617/p4kreviews.csv',encoding='latin-1', index_col=0 )
pitchfork_df
pitchfork_df = pitchfork_df[["review", "score"]]
pitchfork_df.info()
pitchfork_df.dropna(inplace=True)
pitchfork_df.info()
import nltk

import re

stopwords = list(nltk.corpus.stopwords.words('english'))



def preprocessing(text):

    regex = re.compile('[^a-zA-Z\s]+')

    text = text.lower()

    text = regex.sub('', text)

    

    words = text.split()

    words = [word for word in words if word not in stopwords]

    

    return ' '.join(words)

    
pitchfork_df["review"] = pitchfork_df.review.apply(preprocessing)
pitchfork_df
import matplotlib.pyplot as plt



plt.hist(pitchfork_df.score)

plt.show()
# Criar 10 bins para estratificar coluna dos scores (ref: https://michaeljsanders.com/2017/03/24/stratify-continuous-variable.html)

bins = np.linspace(0, len(pitchfork_df), 10)

y_binned = np.digitize(pitchfork_df.score, bins)
from sklearn.model_selection import train_test_split



X, y = pitchfork_df.review, pitchfork_df.score



X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y_binned, random_state=42)
from sklearn.feature_extraction.text import HashingVectorizer

from xgboost import XGBRegressor



cls = XGBRegressor().fit(HashingVectorizer(n_features=2**8).fit_transform(X_train), y_train)

y_pred = cls.predict(HashingVectorizer(n_features=2**8).fit_transform(X_test))
from sklearn.metrics import r2_score



r2_score(y_test, y_pred)