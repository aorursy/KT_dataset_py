import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import librosa

import os

os.chdir("/kaggle/input/train/audio")

genel_yol="/kaggle/input/train/audio/"

os.listdir()

#Audio içindeki seslere göre klasörler.
import IPython.display as ipd

ipd.Audio(genel_yol+"/yes/"+"df1d5024_nohash_3.wav")
x, sr=librosa.load(genel_yol+"/yes/"+"df1d5024_nohash_3.wav")

print(x.shape,sr)
import librosa.display

plt.figure(figsize=(10,5))

librosa.display.waveplot(x,sr=sr);
zero_cross=librosa.feature.zero_crossing_rate(x)

print("Zero Crossing Rate: ",np.mean(zero_cross))
spec_centroid=librosa.feature.spectral_centroid(x,sr=sr)[0]

print(np.mean(spec_centroid))
spec_rolloff=librosa.feature.spectral_rolloff(x,sr=sr)

np.mean(spec_rolloff)
mfk=librosa.feature.mfcc(x,sr=sr)

mfk.shape
genel_yol="/kaggle/input/train/audio/"

liste=[]

turler=[]

adım=0

for tur in os.listdir(genel_yol)[0:2]:#dog ve left

    for ses in os.listdir(genel_yol+tur):

        x,sr=librosa.load(genel_yol+tur+"/"+ses,duration=30)

        liste.append([np.mean(i) for i in librosa.feature.mfcc(x,sr=sr)])

        liste[adım].append(np.mean(librosa.feature.zero_crossing_rate(x)))

        liste[adım].append(np.mean(librosa.feature.spectral_centroid(x,sr=sr)))

        liste[adım].append(np.mean(librosa.feature.spectral_rolloff(x,sr=sr)))                

        adım+=1

        turler.append(tur) #Ozelliklerini çıkardığımız sesin etiketini ayrı bir listeye aynı sırada ekliyoruz.
ozellikler=pd.DataFrame(np.array(liste),index=None)

print(ozellikler.shape)

a=["mfcc"+str(i) for i in range(20)]

a.append("zero_crossing")

a.append("spec_centroid")

a.append("spec_rolloff")

ozellikler.columns=a

ozellikler.head()
turler=pd.DataFrame(turler,columns={"turler"},index=None)

turler.turler.unique()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder().fit(turler)

turler_le=le.transform(turler)

turler_le
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(ozellikler,turler_le, test_size=0.25,random_state=31)
from sklearn.ensemble import RandomForestClassifier

random=RandomForestClassifier().fit(X_train,y_train)

random

from sklearn.metrics import accuracy_score

y_pred=random.predict(X_test)

accuracy_score(y_test,y_pred)