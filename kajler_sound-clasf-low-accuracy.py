import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import librosa

import IPython.display as ipd



import os

print(os.listdir("../input"))
ornek_ses="../input/train/Train/0.wav"

ipd.Audio(ornek_ses)
x,sr=librosa.load(ornek_ses)

print("x Uzunluk:",len(x),"\nKHz:",sr)
mfk=librosa.feature.mfcc(x)

#Daha iyi görmek için DataFrame yaptım.

mfk_dataframe=pd.DataFrame(mfk)

print(mfk_dataframe.shape)

mfk_dataframe.head()
mfk_mean=[np.mean(i) for i in mfk]

print(mfk_mean)
ses_listesi=os.listdir("../input/train/Train") 

ses_listesi.sort()

ses_listesi[0:5]
sınıflar=pd.read_csv("../input/train.csv")

sınıflar.head()
#!pip install soundfile
import soundfile as sf

x,sr=librosa.load(ornek_ses)

print("Librosa Kütüphanesi ile:",x.shape)

x,sr=sf.read(ornek_ses)

print("Soundfile Kütüphanesi ile:",x.shape)
genel_yol="../input/train/Train/"

def mfk_hesap(ses):

    x,sr=sf.read(genel_yol+ses)

    x=parse_audio(x)

    return [np.mean(i) for i in librosa.feature.mfcc(x)]

def parse_audio(x):

    return x.flatten('F')[:x.shape[0]] 
mfk_liste=[]

for ses in ses_listesi:

    mfk_liste.append(mfk_hesap(ses))

dataset=pd.DataFrame(np.array(mfk_liste),index=None)

dataset.head()
X,y=dataset,sınıflar["Class"]
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

le.fit(y)



y = le.transform(y)

y[0:100]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(X_train)



X_train_scaled=scaler.transform(X_train)

X_test_scaled=scaler.transform(X_test)
mlp=MLPClassifier()

model_mlp=mlp.fit(X,y)

model_mlp
from sklearn.metrics import accuracy_score

y_pred=model_mlp.predict(X_test_scaled)

accuracy=accuracy_score(y_test,y_pred)

print("Accuracy Score:",accuracy)
from sklearn.ensemble import GradientBoostingClassifier

gbm_model=GradientBoostingClassifier().fit(X_train_scaled,y_train)

y_pred=gbm_model.predict(X_test)

accuracy_score(y_test,y_pred)