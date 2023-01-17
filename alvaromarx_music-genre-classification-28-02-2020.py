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
GENRES = ['rock','blues','country']



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

    ''' 

    Generates features from audio 

    MFCC (12 coefficients)

    Spectral Centroid 

    RMS

    Spectral Rolloff

    Flatness

    Delta (1st and 2nd order of each feature)

    '''

    labels = range(len(musics))

    features1d = {feature.spectral_centroid: False,  

                  feature.rms: False, 

                  feature.spectral_flatness: False, 

                  feature.mfcc: True}

    f_size=len(features1d)*2*3+1

    feature_array = np.zeros(f_size).reshape(1,f_size)

    for i, genre in enumerate(musics):

        for music, samplerate in genre:

            x = np.array([])

            for feat in features1d.keys():

                if features1d[feat]:

                    f = feat(music, sr=samplerate)

                else:

                    f = feat(music)

                f_delta = feature.delta(f) 

                f_2delta = feature.delta(f, order=2)

                x = np.hstack([x,np.array([f.mean(), np.std(f), f_delta.mean(), np.std(f_delta), f_2delta.mean(), np.std(f_2delta)])])

            x = np.hstack([x,i])

            feature_array = np.vstack([feature_array, x])

#     columns = []

#     for feature in feature1d.keys(): 

#         columns += [feature.__name__ +' mean', feature.__name__ +' std','delta '+feature.__name__ +' mean', ]

#         columns.append(feature.__name__+' mean')

#         columns.append(feature.__name__+' std')

    return pd.DataFrame(data=feature_array).drop(0)#,columns=list(map(lambda x: x.__name__, features1d.keys())))



df_sound = generate_audio_features()

df_sound
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
scaler = StandardScaler()
X,y = df_sound.values[:,:-1], df_sound.values[:,-1]

scaler.fit(X)

X = scaler.transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.decomposition import PCA

import pylab
MIN, MAX, STEP = 5, 15, 2

models = [KNeighborsClassifier(n_neighbors=i) for i in range(MIN,MAX,STEP)]

for i, model in enumerate(models):

    models[i].fit(X_train,y_train)

from sklearn.svm import SVC



plt.plot(range(MIN,MAX,STEP),[model.score(X_test, y_test) for model in models])

plt.xlabel("K Nearest Neighbors")

plt.ylabel("Score")


model = SVC(C=9,probability=True)

model.fit(X_train,y_train)

print(i,model.score(X_test,y_test))
pca = PCA(2)

pca.fit(X_train)

trans_pca = pca.transform(X_test)



for i, _ in enumerate(GENRES):

    pylab.scatter(trans_pca[:,0][y_test==i], trans_pca[:,1][y_test==i],cmap='jet',label=GENRES[i])

pylab.xlabel("PC1")

pylab.ylabel("PC2")

pylab.legend()

pylab.show()

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2,perplexity=10)

X_ = tsne.fit_transform(X_test)

for i, _ in enumerate(GENRES):

    pylab.scatter(X_[:,0][y_test==i], X_[:,1][y_test==i],cmap='jet',label=GENRES[i])

pylab.legend()

pylab.show()
giant_steps, sr_g = sf.read("/kaggle/input/musica2/Giant Steps.wav")

kendrick, sr_k = sf.read("/kaggle/input/musica2/Kendrick Lamar - For Free.wav")
j_music = np.asfortranarray(giant_steps[:sr_g*30,0])

k_music = np.asfortranarray(kendrick[sr_k*60:sr_k*90,0]) 



def generate_audio_features_from(music=j_music, samplerate = sr_g)->pd.DataFrame:

    ''' 

    Generates features from audio 

    MFCC (12 coefficients)

    Spectral Centroid 

    RMS

    Spectral Rolloff

    Flatness

    Delta (1st and 2nd order of each feature)

    '''

    features1d = {feature.spectral_centroid: False,  

                  feature.rms: False, 

                  feature.spectral_flatness: False, 

                  feature.mfcc: True}

    f_size=len(features1d)*2*3

    feature_array = np.zeros(f_size).reshape(1,f_size)

    x = np.array([])

    for feat in features1d.keys():

        if features1d[feat]:

            f = feat(music, sr=samplerate)

        else:

            f = feat(music)

        f_delta = feature.delta(f) 

        f_2delta = feature.delta(f, order=2)

        x = np.hstack([x,np.array([f.mean(), np.std(f), f_delta.mean(), np.std(f_delta), f_2delta.mean(), np.std(f_2delta)])])

    feature_array = np.vstack([feature_array, x])

    return pd.DataFrame(data=feature_array).drop(0)#,columns=list(map(lambda x: x.__name__, features1d.keys())))



feats_k = generate_audio_features_from(k_music, sr_k)

feats_j = generate_audio_features_from(j_music, sr_g)

feats_k_scaled = scaler.transform(feats_k.values)

feats_j_scaled = scaler.transform(feats_j.values)

feats_k = pca.transform(feats_k_scaled)

feats_j = pca.transform(feats_j_scaled)



pca = PCA(2)

pca.fit(X_train)

trans_pca = pca.transform(X_test)



for i, _ in enumerate(GENRES):

    pylab.scatter(trans_pca[:,0][y_test==i], trans_pca[:,1][y_test==i],cmap='jet',label=GENRES[i])

# print(feats_k)

# pylab.scatter(feats_j[0][0],feats_j[0][1],c='r',label='Giant Steps - John Coltrane')



# pylab.scatter(feats_k[0][0],feats_k[0][1],c='g',label='Kendrick Lamar - For Free?')

pylab.xlabel("PC1")

pylab.ylabel("PC2")

pylab.legend()

pylab.show()
sweet_home, sr_alabama = sf.read("/kaggle/input/sweethomealabama/Lynyrd-Skynyrd-Sweet-Home-Alabama.wav")

div = np.array([sweet_home[int(i*sweet_home.shape[0]/30):int((i+1)*sweet_home.shape[0]/30)] for i in range(29)])

div
# model.predict(scaler.transform(generate_audio_features_from(div,sr_alabama)))



feats_sha = [generate_audio_features_from(np.asfortranarray(div[i][:,0]),sr_alabama) for i,_ in enumerate(div)]

for feat in feats_sha:

    print(feat)
feats_sha_scal = scaler.transform(np.array(feats_sha))

feats_sha_scal
print(model.predict_proba(np.array([feats_sha_scal[0]])))

model.predict_proba(np.array([feats_sha_scal[1]]))