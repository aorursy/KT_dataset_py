# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd

import librosa

import librosa.display

import os
GENRES = ['classical','hiphop']

num_lines = sum(1 for line in open('/kaggle/input/audio-path/audio_path.txt'))

num_lines



rotulos = []



file = open("/kaggle/input/audio-path/audio_path.txt", "r")

paths = []

for i, line in enumerate(file):

    

    for j, genre in enumerate(GENRES):

        if genre in line:

            rotulos.append(j)

            if (i < num_lines -1):

                paths.append(line[:-1])

            else:

               paths.append(line) 

    

paths = np.asarray(paths)

rotulos = np.asarray(rotulos)



def import_signal(path):

    s, sr = librosa.core.load(path)

    # slice em 660000 pois é o minimo de todos os audios

    return s[:660000]





signals = []



for p in paths:

    signals.append(import_signal(p))



signals = np.asarray(signals)

# return magnitude S

def stft(signal):

    S, phase = librosa.magphase(np.abs(librosa.stft(signal, hop_length=1024)))

    return S



signals_stft = []

for s in signals:

    signals_stft.append(stft(s))



signals_stft = np.asarray(signals_stft)

signals_stft.shape



def get_features(signals_stft, rotulos):

    def get_centroid(S):

        return librosa.feature.spectral_centroid(S=S)

    def get_flatness(S):

        return librosa.feature.spectral_flatness(S=S)

    def get_rms(s):

        return librosa.feature.rms(s, hop_length=1024)



    info = {'Centroid Mean':[], 

        'Centroid STD': [], 

        'Flatness Mean':[],

        'Flatness STD':[],

        'RMS':[],

        'Target': []} 



    

    for s, rotulo in zip(signals_stft, rotulos):

      

        info['Target'].append(rotulo)

        

        '''

            Obtendo centroide, flatness e RMS

        '''

        c = get_centroid(s)

        c = c[0]

        info['Centroid Mean'].append(np.mean(c))

        info['Centroid STD'].append(np.std(c))

        

        

        f = get_flatness(s)

        f = f[0]

        info['Flatness Mean'].append(np.mean(f))

        info['Flatness STD'].append(np.std(f))

        

        r = get_rms(s)

        r = round(r[0][0],3)

        info['RMS'].append(r)

        

        

        

    return pd.DataFrame(info)
df = get_features(signals_stft, rotulos)

df
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



X,y = df.iloc[:,0:-1], df['Target'].values

scaler.fit(X)

X = scaler.transform(X)
# splitting the data into training and test sets (80:20)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



print(X_train[0])

print(y_train[0])



from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
#Try running from k=1 through 30 and record testing accuracy

k_range = range(1,31)

scores = {}

scores_list = []

for k in k_range:

        knn = KNeighborsClassifier(n_neighbors = k)

        knn.fit(X_train,y_train)

        y_pred = knn.predict(X_test)

        scores[k] = metrics.accuracy_score(y_test,y_pred)

        scores_list.append(metrics.accuracy_score(y_test,y_pred))



print(scores)

def plot_scores(x,y):

    fig, a = plt.subplots(1, figsize = (10, 8))

    title = "Accuracy Score by K values"

    plt.title(title)

    plt.xlabel('Value of K for KNN')

    plt.ylabel('Testing Accuracy')

    a.plot(x, y)

    

plot_scores(list(k_range), scores_list)
k = max(scores, key= scores.get)

print(k)

knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print(y_pred)

metrics.accuracy_score(y_test,y_pred)
from sklearn.decomposition import PCA

import pylab
pca1 = PCA(2)

trans_pca1 = pca1.fit_transform(X_test)

print(pca1.explained_variance_ratio_)



pca2 = PCA(2)

pca2.fit(X_train)

trans_pca2 = pca2.transform(X_test)

print(pca2.explained_variance_ratio_)





for i, _ in enumerate(GENRES):

    pylab.scatter(trans_pca1[:,0][y_test==i], trans_pca1[:,1][y_test==i],cmap='jet',label=GENRES[i])

pylab.xlabel("PC1")

pylab.ylabel("PC2")

pylab.legend()

pylab.show()



for i, _ in enumerate(GENRES):

    pylab.scatter(trans_pca2[:,0][y_test==i], trans_pca2[:,1][y_test==i],cmap='jet',label=GENRES[i])

pylab.xlabel("PC1")

pylab.ylabel("PC2")

pylab.legend()

pylab.show()





pca = PCA(2)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)



knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print(y_pred)

metrics.accuracy_score(y_test,y_pred)
import seaborn as sns
trans_pca1 = pd.DataFrame(trans_pca1)

trans_pca1['Genres'] = [g for y in y_test for i, g in enumerate(GENRES) if y==i]

trans_pca1.columns = ['PC1', 'PC2', 'Genres']

trans_pca1.head()

sns.scatterplot(x=trans_pca1['PC1'], y=trans_pca1['PC2'], hue=trans_pca1['Genres'])