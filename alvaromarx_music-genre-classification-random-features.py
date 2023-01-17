# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from librosa import feature

import matplotlib.pyplot as plt

import soundfile as sf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        break

# Any results you write to the current directory are saved as output.
GENRES = ['classical','hiphop']



def generate_audio_values(genres:list=GENRES,PATH_STRING:str="/kaggle/input/gtzan-genre-collection/genres/")->list:    

    musics = []

    for genre in genres:

        g = []

        for filename in os.listdir(f"{PATH_STRING}{genre}"):

            g.append(sf.read(os.path.join(f"{PATH_STRING}{genre}",filename)))

        musics.append(g)

    return musics



musics = generate_audio_values()


def generate_audio_features(musics:list=musics)->pd.DataFrame:



    N = 128

    K = 12



    global RAND_MAT

    RAND_MAT = np.random.rand(K,N)

    feats = np.zeros((1,2*K+1))

    for i, genre in enumerate(musics):

        for music, samplerate in genre:

            mel = feature.melspectrogram(y=music,sr=samplerate)

            

            dot = np.dot(RAND_MAT, mel) 

            random_features = np.array([i])

            for feat in dot:

                random_features = np.hstack([np.array([feat.mean(),np.std(feat)]),random_features])

            random_features = np.array(random_features)



            feats = np.vstack([feats,random_features])



    return pd.DataFrame(data=feats).drop(0).reset_index(drop=True)

df_sound = generate_audio_features()

df_sound
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
scaler = StandardScaler()
X,y = df_sound.values[:,:-1], df_sound.values[:,-1]

# scaler.fit(X)

# X = scaler.transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.decomposition import PCA

import pylab

from sklearn.naive_bayes import GaussianNB
MIN, MAX, STEP = 1, 16, 2

models = [KNeighborsClassifier(n_neighbors=i) for i in range(MIN,MAX,STEP)]

for i, model in enumerate(models):

    models[i].fit(X_train,y_train)





plt.plot(range(MIN,MAX,STEP),[model.score(X_test, y_test) for model in models])

plt.xlabel("K Nearest Neighbors")

plt.ylabel("Score")

model = GaussianNB()

model.fit(X_train, y_train)

model.score(X_test, y_test)
from sklearn.svm import SVC

svc = SVC(C=100)

svc.fit(X_train, y_train)

svc.score(X_test, y_test)
import dill

dill.dump(svc,open('model_svc_1', 'wb'))
pca = PCA(2)

pca.fit(X_train)

trans_pca = pca.transform(X_test)



for i, _ in enumerate(GENRES):

    pylab.scatter(trans_pca[:,0][y_test==i], trans_pca[:,1][y_test==i],cmap='jet',label=GENRES[i])

pylab.xlabel("PC1")

pylab.ylabel("PC2")

pylab.legend()

pylab.show()

print(pca.explained_variance_ratio_.sum())

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2,perplexity=10)

X_ = tsne.fit_transform(X_test)

for i, _ in enumerate(GENRES):

    pylab.scatter(X_[:,0][y_test==i], X_[:,1][y_test==i],cmap='jet',label=GENRES[i])

pylab.legend()

pylab.show()

