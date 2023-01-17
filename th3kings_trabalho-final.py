# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import soundfile as sf

import librosa

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

import IPython.display as ipd



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sound, samplerate = sf.read("/kaggle/input/linkin-park/In_The_End-Linkin_Park.wav")

sound = sound[:,0]



w_size = sound.shape[0]//samplerate

window = np.array_split(sound, w_size)



features = np.zeros(14)

for i in range(w_size):

    

    window = np.asfortranarray(sound[i*samplerate:(i+1)*samplerate])

    flatness = librosa.feature.spectral_flatness(y=window)

    rms = librosa.feature.rms(window)

    rolloff = librosa.feature.spectral_rolloff(window)

    mfcc = librosa.feature.mfcc(y=window,sr=samplerate,n_mfcc=5)

    mfcc = mfcc[1:,:]

    

    f = np.hstack([flatness.mean(axis=1), flatness.std(axis=1), rms.mean(axis=1), rms.std(axis=1), rolloff.mean(axis=1), rolloff.std(axis=1), mfcc.mean(axis=1), mfcc.std(axis=1)])

    features = np.vstack([features,f])

    

features = features[1:,:]

scl = StandardScaler()

scl.fit(features)

x = scl.transform(features)





inertia = []

clusters = range(2,20)

for n_c in clusters:

    model = KMeans(n_clusters=n_c, random_state=42)

    model.fit(x)

    inertia.append(model.inertia_)



plt.plot(clusters, inertia)
clusteres = 5

model = KMeans(n_clusters=clusteres, random_state=42)

model.fit(x)

labels  = model.predict(x)

plt.plot(labels)
clusters = model.transform(x)

P = (1/(10**-6 + clusters)) / np.sum(1/(10**-6 + clusters), axis=1, keepdims=1)

def transition_matrix(ndim, p_stay):

  T = np.ones ( (ndim, ndim)) * ((1-p_stay)/(ndim-1))

  T *= 1-np.eye(ndim)

  T += np.eye(ndim)*p_stay

  return T

T = transition_matrix(clusteres, .9)

states = librosa.sequence.viterbi(P.T, T)

plt.plot(states)
def get_soundfile(file, genre):

    return sf.read(f"/kaggle/input/gtzan-genre-collection/genres/{genre}/{file}")



df = pd.read_csv("/kaggle/input/trabalho-final-csv/sound.csv")

rock=[]

srt_rock=[]

hiphop=[]

srt_hiphop=[]





for i in range(len(df["file"])):



    x, y = get_soundfile(df["file"][i] , df["genre"][i])

        

    if(df["genre"][i] == "rock"):

        rock.append(x)

        srt_rock.append(y)

        

    else:

        hiphop.append(x)

        srt_hiphop.append(y)

        

rms_rock=[]

rms_hiphop=[]

centroid_rock=[]

centroid_hiphop=[]

flatness_rock=[]

flatness_hiphop=[]



for i in range(len(rock)):

    rms_rock.append(librosa.feature.rms(y=rock[i]))

    rms_hiphop.append(librosa.feature.rms(y=hiphop[i]))

    centroid_rock.append(librosa.feature.spectral_centroid(y=rock[i]))

    centroid_hiphop.append(librosa.feature.spectral_centroid(y=hiphop[i]))

    flatness_rock.append(librosa.feature.spectral_flatness(y=rock[i]))

    flatness_hiphop.append(librosa.feature.spectral_flatness(y=hiphop[i]))

    

fts = np.zeros((1,7))



for i in range(len(rock)):

    f = np.hstack([rms_rock[i].max(), rms_rock[i].std(), centroid_rock[i].max(), centroid_rock[i].std(), flatness_rock[i].min(), flatness_rock[i].std(), "rock"])

    fts=np.vstack([fts,f])

    

for i in range(len(hiphop)):

    f = np.hstack([rms_hiphop[i].max(), rms_hiphop[i].std(), centroid_hiphop[i].max(), centroid_hiphop[i].std(), flatness_hiphop[i].min(), flatness_hiphop[i].std(), "hiphop"])

    fts=np.vstack([fts,f])

    

fts = fts[1:,:]

Df = pd.DataFrame(data=fts)

Df.head()
x, y = Df.iloc[:,:-1], Df.iloc[:,-1]

y, labels = pd.factorize(y)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 67)



scaler = StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)



train_scores = np.zeros(0)

test_scores = np.zeros(0)

knns = []

for i in range(1,10):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    test_scores = np.hstack([test_scores, knn.score(x_test,y_test)])

    knns.append(knn)



plt.plot(range(1,10),test_scores,label="test score")

plt.xlabel('Número de Vizinhos')

plt.ylabel('Taxa de Acertos')

plt.legend()

plt.show()
knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(x,y)
sound, samplerate = librosa.load("/kaggle/input/linkin-park/In_The_End-Linkin_Park.wav")

changes = np.where(states[:-1] != states[1:])[0]

musica = []



for i in range(int(len(sound)/samplerate)):

    musica.append(sound[(i-1)*samplerate:(i*samplerate)])

musica.pop(0)



if (changes[0]==0):

    changes = np.delete(changes,0)



seg_musica = []

start=0



for i in changes:

    aux=np.zeros(0)

    for j in range(start,i):

        aux=np.hstack([aux,musica[j]])

    start=i

    seg_musica.append(aux)

    

previsao=[]

    

for trecho in seg_musica:



    rms_trecho = librosa.feature.rms(y=trecho)

    centroid_trecho=librosa.feature.spectral_centroid(y=trecho)

    flatness_trecho=librosa.feature.spectral_flatness(y=trecho)

    

    fts_trecho=[rms_trecho.max(), rms_trecho.std(), centroid_trecho.max(), centroid_trecho.std(), flatness_trecho.min(), flatness_trecho.std()]

    Df_trecho = pd.DataFrame([fts_trecho])

    

    previsao.append(knn.predict_proba(Df_trecho)[0])



chance_rock=0

chance_hiphop=0



for i in range(len(previsao)):

    chance_rock += previsao[i][0]

    chance_hiphop += previsao[i][1]

    

print("A chance da musica ser um rock eh de: "+ str(100*(chance_rock/len(previsao))) +"%")

print("A chance da musica ser um hiphop eh de: "+ str(100*(chance_hiphop/len(previsao))) +"%")
