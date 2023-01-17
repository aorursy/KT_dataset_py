# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import librosa
from librosa import display as dsp
from matplotlib import pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential

# Any results you write to the current directory are saved as output.
soundFilesList=os.listdir('../input/GuitarNotes/GuitarNotes/')
soundLabel=[]
soundData=[]
rawData=[]
baseDir='../input/GuitarNotes/GuitarNotes/'
mfccSingularList=list()
for file in soundFilesList:
    tempAudioData=list()
    audioData,sampleRate=librosa.core.load(baseDir+file,res_type='kaiser_fast')
    rawData.append(audioData)
    spectralData=librosa.feature.spectral_bandwidth(y=audioData,sr=sampleRate)
    for sd in spectralData:
        tempAudioData.append(sd)
    soundData.append(tempAudioData)
    soundLabel.append(str.split(file,'.')[0])
print('Done')
idx=np.random.randint(len(soundData))
plt.plot(rawData[idx])
plt.xlabel(soundLabel[idx])
plt.show()
plt.plot(soundData[idx][0])
plt.grid()
plt.show()
X=[]
for s in soundData:
    for val in s:
        X.append(val)
print(max([len(x) for x in X]))
maxlen=max([len(x) for x in X])
X=pad_sequences(X,maxlen=maxlen)
np.shape(X)
idx1,idx2=np.random.randint(len(X)),np.random.randint(len(X))
plt.scatter(X[idx1],X[idx2],c=['r','b'],alpha=0.5)
plt.xlabel(soundLabel[idx1])
plt.ylabel(soundLabel[idx2])
plt.show()
idx=np.random.randint(len(X))
plt.plot(soundData[idx][0])
plt.ylabel(soundLabel[idx])
plt.grid()
plt.show()
np.shape(X)
num_classes=len(set(soundLabel))
num_classes
targetLabelEncoder=LabelEncoder()
y=targetLabelEncoder.fit_transform(soundLabel)
#X=X.reshape(X.shape[0],X.shape[1],1)
#y=to_categorical(y=target,num_classes=num_classes)
#input_shape=(X.shape[1],1)
np.shape(X),np.shape(y)
from sklearn.svm import SVC
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.80,test_size=0.20,random_state=43)
svcClf=SVC()
svcClf.fit(X,y)
print('Accuracy:',svcClf.score(X_test,y_test))
idx=np.random.randint(len(X_test))
print('Pred:',targetLabelEncoder.inverse_transform(svcClf.predict([X_test[idx]]))[0])
print('Actual:',targetLabelEncoder.inverse_transform(y_test[idx]))