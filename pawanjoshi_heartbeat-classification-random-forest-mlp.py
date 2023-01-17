import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os,fnmatch
music="/kaggle/input/heartbeat-sounds/set_a/"

import librosa

import IPython.display as ipd

x,sr=librosa.load(music+"normal__201106221418.wav",duration=5)

ipd.Audio(x,rate=sr)
def ses_df(music_folders,kolonlar,tür_liste):

    liste=[]

    adim=0

    for folder in music_folders:

        for tür in tür_liste:

            dosyalar=fnmatch.filter(os.listdir(folder),tür)

            label=tür.split("*")[0]

            for dosya in dosyalar:

                x,sr=librosa.load(folder+dosya,duration=5,res_type='kaiser_fast')

                liste.append([np.mean(x) for x in librosa.feature.mfcc(x,sr=sr)])

                liste[adim].append(sum(librosa.zero_crossings(x)))

                liste[adim].append(np.mean(librosa.feature.spectral_centroid(x)))

                liste[adim].append(np.mean(librosa.feature.spectral_rolloff(x,sr=sr)))

                liste[adim].append(np.mean(librosa.feature.chroma_stft(x,sr=sr)))

                liste[adim].append(label)

                liste[adim].append(dosya)

                adim+=1

    return pd.DataFrame(liste,columns=kolonlar)
music_folders=["/kaggle/input/heartbeat-sounds/set_a/","/kaggle/input/heartbeat-sounds/set_b/"]

kolonlar=["mfkk"+str(i) for i in range(20)]

for isim in ["zero","centroid","rolloff","chroma","tür","dosya"]:

    kolonlar.append(isim)

tür_liste=["normal*.wav","artifact*.wav","murmur*.wav"]

music_df=ses_df(music_folders,kolonlar,tür_liste)
print(music_df.shape)

music_df.head()
music_df["tür"].value_counts()
X=music_df.iloc[:,0:24]

X.head()
y=music_df["tür"]

y.head()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder().fit(y)

y=le.transform(y)

y[10:40]
from sklearn.model_selection import train_test_split,GridSearchCV

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=31)

print("X Train: ",len(X_train),"\n","X Test: ",len(X_test),sep="")
from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(max_depth= 8,

 max_features= 5,

 min_samples_split=5,

 n_estimators=500).fit(X_train,y_train)

forest
from sklearn.metrics import accuracy_score

y_pred=forest.predict(X_test)

accuracy_score(y_test,y_pred)
from sklearn.metrics import confusion_matrix

import seaborn as sns

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap="YlGnBu")
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(X_train)



X_train_scaled=scaler.transform(X_train)

X_test_scaled=scaler.transform(X_test)
mlp=MLPClassifier().fit(X_train_scaled,y_train)

mlp
y_pred=mlp.predict(X_test_scaled)

accuracy_score(y_test,y_pred)
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap="YlGnBu")