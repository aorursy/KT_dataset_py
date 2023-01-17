from google.colab import drive
drive.mount('/content/gdrive')
# Importing necessary packages 
import os
import zipfile
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import clear_output

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics

import librosa
import librosa.display as libd

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.models import Model

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
# Unzip speech data
zipsrc=zipfile.ZipFile("/content/gdrive/My Drive/iss/rtavs/data/speechsub.zip",'r')
zipsrc.extractall("/content")
zipsrc.close()
print("Unzip completed..")


# Setting Current working directory
os.chdir('/content')
print("current working directory")
os.getcwd()
# Set Matplotlib Default styles 
plt.style.use('seaborn')
plt.rcParams['ytick.right']=True
plt.rcParams['ytick.labelright']=True
plt.rcParams['ytick.left']=False
plt.rcParams['ytick.labelleft']=False
plt.rcParams['figure.figsize']=[7,7]
print('Default style setup completes..')
audiopath="speechsub"
labels=[dirname[1] for dirname in os.walk(audiopath)]
labels=labels[0]
labels
nosample=os.path.join(audiopath,'no','0c2ca723_nohash_0.wav')
smp,smpR=librosa.load(nosample,sr=16000)
print(smp.shape)
plt.figure(figsize=(8,4))
plt.plot(np.linspace(0,len(smp)/smpR,len(smp)),smp)
plt.title('Sound of "No" ..')
plt.xlabel("Time in seconds ")
plt.ylabel("amplitude")
plt.show()
# Checking no of audio file for each label and draw braw chart to see balanced data or Imbalance data
allrecordscount=[]
totalRecords=0
for lbl in labels:
  pth=os.path.join(audiopath,lbl)
  records=[f for f in os.listdir(pth) if f.endswith(".wav")]
  allrecordscount.append(len(records))
  totalRecords    = totalRecords+len(records)
plt.figure(figsize=(10,5))
plt.bar(labels,allrecordscount,color='C2')

# checking audio duration for all files and plot histogram 
durations=[]

for lbl in labels:
  pth=os.path.join(audiopath,lbl)
  allfiles =[f for f in os.listdir(pth) if f.endswith('.wav')]
  for fil in allfiles:
    smr,sm=wavfile.read(os.path.join(pth,fil))
    durations.append(len(sm)/smr)
  
plt.figure(figsize=(8,8))
plt.hist(durations)
plt.title("distribution of duration")
plt.ylabel('No of records')
plt.xlabel('Time in seconds ')

fftSize = 512
smpStft = np.abs(librosa.stft(y=smp,n_fft=fftSize))
spectogram = librosa.amplitude_to_db(smpStft,ref=np.max)
print(spectogram.shape)
plt.figure()
libd.specshow(spectogram,sr=16000,hop_length=fftSize/4,y_axis='log',x_axis='time')
plt.title('Spectogram of the "No" ...',fontsize=10)
plt.xlabel('time (s)')
plt.colorbar(format='%+2.0f dB')
smpMfcc = librosa.feature.mfcc(y=smp,sr=16000,n_mfcc=40)
smpMfcc = sklearn.preprocessing.scale(smpMfcc,axis=1)
print(smpMfcc.shape)
print(smpMfcc)
plt.figure()
libd.specshow(smpMfcc,sr=16000,hop_length=512,x_axis='time')
plt.title('Spectogram of the "go" ...',fontsize=10)
plt.colorbar()
allRecordsspec=[]
allRecordsmfcc=[]
allLabels=[]
resampRate=8000
inputLength=8000
start=timeit.default_timer()
run=1
for lbl in labels:
  pth=os.path.join(audiopath,lbl)
  files=[f for f in os.listdir(pth) if f.endswith('.wav') ]
  for fil in files:
    smp,smpR=librosa.load(os.path.join(pth,fil),sr=16000)
    smp=librosa.resample(smp,smpR,resampRate)
    fftSize = 512
    smpStft = np.abs(librosa.stft(y=smp,n_fft=fftSize))
    spectogram = librosa.amplitude_to_db(smpStft,ref=np.max)
    smpMfcc = librosa.feature.mfcc(y=smp,sr=16000,n_mfcc=40)
    smpMfcc = sklearn.preprocessing.scale(smpMfcc,axis=1)
    print(len(smp))
    if (len(smp)==inputLength):
      allRecordsspec.append(spectogram)
      allRecordsmfcc.append(smpMfcc)
      allLabels.append(lbl)
    clear_output(wait=True)
    stop=timeit.default_timer()
    if (run/totalRecords)<0.05:
      timeExpected  = "Calculating ..."
    else:
      timenow=timeit.default_timer()
      timeexpected=np.round((timenow-start)/run*totalRecords/60,2)
    print("Checking progress:", run ,"records")
    print("Time taken       : ", np.round((stop-start)/60,2), "minutes")
    print("Expected duration:", timeExpected, "minutes")
    print('')
    run = run+1
  #allRecordsspec  = np.array(allRecordsspec).reshape(-1,inputLength,1)
  #allRecordsspec  = np.array(allRecordsspec).reshape(-1,inputLength,1)
#print("The shape of allRecords is", allRecordsspec.shape, " allRecordsmfcc shape:- ", allRecordsmfcc.shape, "and the data type is", allRecordsmfcc.dtype) 
np.save("/content/gdrive/My Drive/iss/rtavs/data/allRV1Records_sep.npy",allRecordsspec)
np.save("/content/gdrive/My Drive/iss/rtavs/data/allRV1Records_mfcc.npy",allRecordsmfcc)
np.save("/content/gdrive/My Drive/iss/rtavs/data/allRV1Labels.npy",allLabels)

np.save("/content/gdrive/My Drive/iss/rtavs/data/allRV1Records_sep.npy",allRecordsspec)
np.save("/content/gdrive/My Drive/iss/rtavs/data/allRV1Records_mfcc.npy",allRecordsmfcc)
np.save("/content/gdrive/My Drive/iss/rtavs/data/allRV1Labels.npy",allLabels)

spect()
# Preparing Data Set 
allRecords=[]
allLabels=[]
resampRate=8000
inputLength=8000
start=timeit.default_timer()
run=1
for lbl in labels:
  pth=os.path.join(audiopath,lbl)
  files=[f for f in os.listdir(pth) if f.endswith('.wav') ]
  for fil in files:
    smp,smpR=librosa.load(os.path.join(pth,fil),sr=16000)
    smp=librosa.resample(smp,smpR,resampRate)
    if (len(smp)==inputLength):
      allRecords.append(smp)
      allLabels.append(lbl)
    clear_output(wait=True)
    stop=timeit.default_timer()
    if (run/totalRecords)<0.05:
      timeExpected  = "Calculating ..."
    else:
      timenow=timeit.default_timer()
      timeexpected=np.round((timenow-start)/run*totalRecords/60,2)
    print("Checking progress:", run ,"records")
    print("Time taken       : ", np.round((stop-start)/60,2), "minutes")
    print("Expected duration:", timeExpected, "minutes")
    print('')
    run = run+1
allRecords  = np.array(allRecords).reshape(-1,inputLength,1)
print("The shape of allRecords is", allRecords.shape, "and the data type is", allRecords.dtype) 
np.save("/content/gdrive/My Drive/iss/rtavs/data/allRV1Records.npy",allRecords)
np.save("/content/gdrive/My Drive/iss/rtavs/data/allRV1Labels.npy",allLabels)

allLabels=np.load("/content/gdrive/My Drive/iss/rtavs/data/allRV1Labels.npy")
allRecords=np.load("/content/gdrive/My Drive/iss/rtavs/data/allRV1Records.npy")
le=LabelEncoder()
lbls=le.fit_transform(allLabels)
classes=list(le.classes_)
classes=[str(c) for c in classes]
lbls=to_categorical(lbls,num_classes=len(classes))
  
# Split Data set into traing and test 

(trDat,vlDat,trLbl,vlLbl)= train_test_split(allRecords,lbls,stratify=lbls,test_size=0.2,
                                                  random_state=229,shuffle=True)
trDat.shape
modelname="speechRV1"
def createmodel(inputSize):
  ipt=Input(shape=(inputSize,1))
  x=Conv1D(8,11,padding="valid",activation="relu")(ipt)
  x=MaxPool1D(4)(x)
  x=Dropout(0.25)(x)

  x=Conv1D(16,11,padding="valid",activation="relu")(x)
  x=MaxPool1D(4)(x)
  x=Dropout(0.25)(x)

  x=Conv1D(32,11,padding="valid",activation="relu")(x)
  x=MaxPool1D(4)(x)
  x=Dropout(0.25)(x)

  x=Conv1D(64,11,padding="valid",activation="relu")(x)
  x=MaxPool1D(4)(x)
  x=Dropout(0.25)(x)

  x=Flatten()(x)

  x=Dense(256,activation="relu")(x)
  x=Dropout(0.25)(x)

  x=Dense(128,activation="relu")(x)
  x=Dropout(0.25)(x)

  x=Dense(10,activation="softmax")(x)

  model=Model(ipt,x)
  model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
  return model
inputLength=8000
folderpath = '/content/gdrive/My Drive/iss/rtavs/colab/'
model=createmodel(inputLength)
modelGo=createmodel(inputLength)
modelGo.save(folderpath+"speechRV1Model.pb")
modelGo.summary()
plot_model(model)
# Create Call Backs 

folderpath      = '/content/gdrive/My Drive/iss/rtavs/colab/'
filepath        = folderpath + modelname + ".hdf5"
checkpoint=ModelCheckpoint(filepath,monitor="val_accuracy",verbose=0,save_best_only=True,mode="max")
csvlogger=CSVLogger(folderpath+modelname+'.csv')
callbacks_list=[checkpoint,csvlogger]
print("Callbacks created:")
print(callbacks_list[0])
print(callbacks_list[1])
print('')
print("Path to model:", filepath)
print("Path to log:  ", folderpath+modelname+'.csv')
#model.fit(trDat,trLbl,validation_data=(vlDat,vlLbl),epochs=100,batch_size=32,shuffle=True,callbacks=callbacks_list)
folderpath      = '/content/gdrive/My Drive/iss/rtavs/colab/'
filepath        = folderpath + modelname + ".hdf5"
modelGo.load_weights(filepath)
modelGo.compile(loss='categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
modelGo.predict(vlDat)
predout=np.argmax(predits,axis=1)
testout=np.argmax(vlLbl,axis=1)
print(predout)
print(testout)
testscore=metrics.accuracy_score(predout,testout)
print("Test score",testscore*100)
print(metrics.classification_report(testout,predout,target_names=classes))
confusion=metrics.confusion_matrix(testout,predout)
print(confusion)
def EncodeLabels(labelpath):
  allLabels=np.load(labelpath)
  le=LabelEncoder()
  lbls=le.fit_transform(allLabels)
  classes=list(le.classes_)
  classes=[str(c) for c in classes]
  return classes
def makeprediction(audiofile,modelfolderpath,labelpath):
  modelpath=modelfolderpath+"speechRV1Model"
  modelweightpath=modelfolderpath+"speechRV1.hdf5"
  resampRate=8000
  smp,smpR=librosa.load(audiofile,sr=16000)
  smp=librosa.resample(smp,smpR,resampRate)
  smp=smp.reshape(-1,inputLength,1)
  modelGo=tf.keras.models.load_model(modelpath)
  modelGo.load_weights(modelweightpath)
  predict=modelGo.predict(smp)
  predict=np.argmax(predict,axis=1)
  classes=EncodeLabels(labelpath)
  return classes[predict[0]]
pred=makeprediction("/content/gdrive/My Drive/iss/rtavs/data/s.wav","/content/gdrive/My Drive/iss/rtavs/colab/",
                    "/content/gdrive/My Drive/iss/rtavs/data/allRV1Labels.npy")
print("Class label :- ",pred)
