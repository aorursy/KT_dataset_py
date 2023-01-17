# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import librosa

import librosa.display

import soundfile as sf

import seaborn as sns

import matplotlib.pyplot as plt

import IPython.display as ipd

# Input data files are availab

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from librosa import feature

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
generos = ['rock','jazz']

def get_sounds(generos,path='/kaggle/input/gtzan-genre-collection/genres/'):

    musicas = []

    for genero in generos:

        lista_generos = []

        for filename in os.listdir(f"{path}{genero}"):

            lista_generos.append(sf.read(os.path.join(f"{path}{genero}",filename)))

        musicas.append(lista_generos)

    return musicas

data = get_sounds(generos)

print('rock = 0 ,jazz=1 ,classical = 2')
def generate_features(musicas):

    labels = range(len(musicas))

    features1d = {feature.spectral_centroid: False,

                  feature.rms: False,

                  feature.spectral_flatness: False,

                  feature.mfcc: False}

    f_size=len(features1d)*2*3+1

    feature_array = np.zeros(f_size).reshape(1,f_size)

    for i,genero in enumerate(musicas):

        for music,samplerate in genero:

            x = np.array([])

            for feat in features1d.keys():

                if features1d[feat]:

                    f = feat(music,sr=samplerate)

                else:

                    f = feat(music)

                f_delta = feature.delta(f)

                f_2delta = feature.delta(f,order=2)

                x = np.hstack([x,np.array([f.mean(), np.std(f), f_delta.mean(), np.std(f_delta), f_2delta.mean(), np.std(f_2delta)])])

            x = np.hstack([x,i])

            feature_array = np.vstack([feature_array,x])

    return pd.DataFrame(data=feature_array).drop(0)

df_sound = generate_features(data)
df_sound.to_csv('./jazz-rock.csv')
X,y = df_sound.iloc[:,:-1], df_sound.iloc[:,-1]

scaler = StandardScaler()

import joblib

scaler.fit(X)

joblib.dump(scaler,'scaler.jbl')

X = scaler.transform(X)

X
pca = PCA(2)

pca.fit(X)

joblib.dump(pca,'pca.jbl')

pca_X = pca.transform(X)

sns.scatterplot(x=pca_X[:,0],y=pca_X[:,1],hue=y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train,y_train)

decision_tree.score(X_test,y_test)
joblib.dump(decision_tree, "modelo.jbl") # salva a variavel

#variavel = joblib.load("variavel.jbl")  # carrega a variável