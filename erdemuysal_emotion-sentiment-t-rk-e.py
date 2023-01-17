import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import librosa

import librosa.display

import glob 

from sklearn.metrics import confusion_matrix

import IPython.display as ipd  # To play sound in the notebook

import os

import sys
ROOT = '/kaggle/input/turkish-speech-dataset/'
emotion=[]

path = []

i = 0



for dirname, _, mutlu_files in os.walk('/kaggle/input/turkish-speech-dataset/mutlu2'):

    for file in os.listdir(dirname):

        emotion.append('mutlu')

        path.append('/kaggle/input/turkish-speech-dataset/mutlu2/' + file) 

    

for dirname, _, normal_files in os.walk('/kaggle/input/turkish-speech-dataset/normal2'):

    for file in os.listdir(dirname):

        emotion.append('normal')

        path.append('/kaggle/input/turkish-speech-dataset/normal2/' + file)



df = pd.DataFrame(emotion, columns = ['labels'])

df = pd.concat([df, pd.DataFrame(path, columns = ['path'])], axis = 1)

df.labels.value_counts()
df
fname = ROOT + 'mutlu2/export (0).wav'  

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Lets play the audio 

ipd.Audio(fname)
fname = ROOT + 'normal2/export (0).wav'

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Lets play the audio 

ipd.Audio(fname)
print(df.labels.value_counts())

df.to_csv('Data_path.csv', index=False)
ref = pd.read_csv('./Data_path.csv')

ref.head()
path = ROOT + 'normal2/export (13).wav'

X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  

mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)



# Audio Wave

plt.figure(figsize=(20, 15))

plt.subplot(3,1,1)

librosa.display.waveplot(X, sr=sample_rate)

plt.title('Audio sampled at 44100 hrz')



# MFCC

plt.figure(figsize=(20, 15))

plt.subplot(3,1,1)

# Here we are displaying Spectrogram for the Happy voice and lets visualiza how its look like

librosa.display.specshow(mfcc, x_axis='time')

plt.ylabel('MFCC')

plt.colorbar()



ipd.Audio(path)
df = pd.DataFrame(columns=['feature'])



# loop feature extraction over the entire dataset

counter=0

for index, path in enumerate(ref.path):

    X, sample_rate = librosa.load(path

                                  ,res_type='kaiser_fast'

                                  ,duration=2.5

                                  ,sr=44100

                                  ,offset=0.5

                                 )

    sample_rate = np.array(sample_rate)

    

    # mean as the feature. Could do min and max etc as well. 

    mfccs = np.mean(librosa.feature.mfcc(y=X, 

                                         sr=sample_rate, 

                                         n_mfcc=13),

                    axis=0)

    

    df.loc[counter] = [mfccs]

    counter=counter+1   



# Check a few records to make sure its processed successfully

print(len(df))

df.head()
#concatinating the feature column into the complete dataframe

df = pd.concat([ref, pd.DataFrame(df['feature'].values.tolist())],axis=1)

df[:5]
# replace NA with 0

df=df.fillna(0)

print(df.shape)

df[:5]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df.drop(['path','labels'],axis=1)

                                                    , df.labels

                                                    , test_size=0.25

                                                    , shuffle=True

                                                    , random_state=42

                                                   )



# Lets see how the data present itself before normalisation 

X_train[10:20]
# Lets do data normalization

#Here we are using z-score normalization technique

mean = np.mean(X_train, axis=0)

std = np.std(X_train, axis=0)



X_train = (X_train - mean)/std

X_test = (X_test - mean)/std



# Check the dataset now 

X_train[10:20]
from keras.utils import np_utils, to_categorical

from sklearn.preprocessing import LabelEncoder

import pickle
X_train = np.array(X_train)

y_train = np.array(y_train)

X_test = np.array(X_test)

y_test = np.array(y_test)



# Label encode the target 

lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))

y_test = np_utils.to_categorical(lb.fit_transform(y_test))



print(X_train.shape)

print(lb.classes_)

#print(y_train[0:10])

#print(y_test[0:10])



# Pickel the lb object for future use 

filename = 'labels'

outfile = open(filename,'wb')

pickle.dump(lb,outfile)

outfile.close()
X_train = np.expand_dims(X_train, axis=2)

X_test = np.expand_dims(X_test, axis=2)

X_train.shape
import keras

from keras import regularizers

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, Model, model_from_json

from keras.layers import Dense, Embedding, LSTM

from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization

from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D

from keras.utils import np_utils, to_categorical

from keras.callbacks import ModelCheckpoint
X_train.shape[1]
model = Sequential()

model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1))) 

# X_train.shape[1] = No. of Columns

model.add(Activation('relu'))

model.add(Conv1D(256, 8, padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(MaxPooling1D(pool_size=(8)))

model.add(Conv1D(128, 8, padding='same'))

model.add(Activation('relu'))

model.add(Conv1D(128, 8, padding='same'))

model.add(Activation('relu'))

model.add(Conv1D(128, 8, padding='same'))

model.add(Activation('relu'))

model.add(Conv1D(128, 8, padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(MaxPooling1D(pool_size=(8)))

model.add(Conv1D(64, 8, padding='same'))

model.add(Activation('relu'))

model.add(Conv1D(64, 8, padding='same'))

model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(2)) # Target class number

model.add(Activation('softmax'))

opt = keras.optimizers.Adam(lr=0.00001)

model.summary()
model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])

model_history=model.fit(X_train, y_train, batch_size=4, epochs=15, validation_data=(X_test, y_test))
plt.plot(model_history.history['loss'])

plt.plot(model_history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# Save model and weights

model_name = 'Emotion_Model.h5'

save_dir = os.path.join(os.getcwd(), 'saved_models')



if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)

model.save(model_path)

print('Save model and weights at %s ' % model_path)



# Save the model to disk

model_json = model.to_json()

with open("model_json.json", "w") as json_file:

    json_file.write(model_json)
# loading json and model architecture 

json_file = open('model_json.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



# load weights into new model

loaded_model.load_weights("saved_models/Emotion_Model.h5")

print("Loaded model from disk")

 

# Keras optimiser

opt = keras.optimizers.Adam(lr=0.0001)

loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

score = loaded_model.evaluate(X_test, y_test, verbose=0)

print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
newData,newSR= librosa.load("../input/turkish-speech-dataset/mutlu2/export (32).wav")

ipd.Audio("../input/turkish-speech-dataset/mutlu2/export (32).wav")
plt.figure(figsize=(15, 5))

librosa.display.waveplot(newData, sr=newSR)



# Lets transform the dataset so we can apply the predictions

newData, newSR = librosa.load("../input/turkish-speech-dataset/mutlu2/export (32).wav"

                              ,duration=2.5

                              ,sr=44100

                              ,offset=0)



newSR = np.array(newSR)

mfccs = np.mean(librosa.feature.mfcc(y=newData, sr=newSR, n_mfcc=13),axis=0)

newdf = pd.DataFrame(data=mfccs).T

print(newdf.shape)



newdf= np.expand_dims(newdf,axis=2)

print(newdf.shape)

newpred=model.predict(newdf)



filename = filename = './labels'

infile = open(filename,'rb')

lb = pickle.load(infile)

infile.close()



# Get the final predicted label

final = newpred.argmax(axis=1)

final = final.astype(int).flatten()

final = (lb.inverse_transform((final)))

print(final) #emo(final) #gender(final) 
newData,newSR= librosa.load("../input/turkish-speech-dataset/mutlu2/export (33).wav")

ipd.Audio("../input/turkish-speech-dataset/mutlu2/export (33).wav")
plt.figure(figsize=(15, 5))

librosa.display.waveplot(newData, sr=newSR)



# Lets transform the dataset so we can apply the predictions

newData, newSR = librosa.load("../input/turkish-speech-dataset/mutlu2/export (33).wav"

                              ,duration=2.5

                              ,sr=44100

                              ,offset=0)



newSR = np.array(newSR)

mfccs = np.mean(librosa.feature.mfcc(y=newData, sr=newSR, n_mfcc=13),axis=0)

newdf = pd.DataFrame(data=mfccs).T

print(newdf.shape)



newdf= np.expand_dims(newdf,axis=2)

print(newdf.shape)

newpred=model.predict(newdf)



filename = filename = './labels'

infile = open(filename,'rb')

lb = pickle.load(infile)

infile.close()



# Get the final predicted label

final = newpred.argmax(axis=1)

final = final.astype(int).flatten()

final = (lb.inverse_transform((final)))

print(final) #emo(final) #gender(final) 
newData,newSR= librosa.load("../input/turkish-speech-dataset/mutlu2/export (51).wav")

ipd.Audio("../input/turkish-speech-dataset/mutlu2/export (51).wav")
plt.figure(figsize=(15, 5))

librosa.display.waveplot(newData, sr=newSR)



# Lets transform the dataset so we can apply the predictions

newData, newSR = librosa.load("../input/turkish-speech-dataset/mutlu2/export (51).wav"

                              ,duration=2.5

                              ,sr=44100

                              ,offset=0)



newSR = np.array(newSR)

mfccs = np.mean(librosa.feature.mfcc(y=newData, sr=newSR, n_mfcc=13),axis=0)

newdf = pd.DataFrame(data=mfccs).T

print(newdf.shape)



newdf= np.expand_dims(newdf,axis=2)

print(newdf.shape)

newpred=model.predict(newdf)



filename = filename = './labels'

infile = open(filename,'rb')

lb = pickle.load(infile)

infile.close()



# Get the final predicted label

final = newpred.argmax(axis=1)

final = final.astype(int).flatten()

final = (lb.inverse_transform((final)))

print(final) #emo(final) #gender(final) 
newData,newSR= librosa.load("../input/turkish-speech-dataset/normal2/export (15).wav")

ipd.Audio("../input/turkish-speech-dataset/normal2/export (15).wav")
plt.figure(figsize=(15, 5))

librosa.display.waveplot(newData, sr=newSR)



# Lets transform the dataset so we can apply the predictions

newData, newSR = librosa.load("../input/turkish-speech-dataset/normal2/export (15).wav"

                              ,duration=2.5

                              ,sr=44100

                              ,offset=0)



newSR = np.array(newSR)

mfccs = np.mean(librosa.feature.mfcc(y=newData, sr=newSR, n_mfcc=13),axis=0)

newdf = pd.DataFrame(data=mfccs).T

print(newdf.shape)



newdf= np.expand_dims(newdf,axis=2)

print(newdf.shape)

newpred=model.predict(newdf)



filename = filename = './labels'

infile = open(filename,'rb')

lb = pickle.load(infile)

infile.close()



# Get the final predicted label

final = newpred.argmax(axis=1)

final = final.astype(int).flatten()

final = (lb.inverse_transform((final)))

print(final) #emo(final) #gender(final) 
newData,newSR= librosa.load("../input/turkish-speech-dataset/normal2/export (46).wav")

ipd.Audio("../input/turkish-speech-dataset/normal2/export (46).wav")
plt.figure(figsize=(15, 5))

librosa.display.waveplot(newData, sr=newSR)



# Lets transform the dataset so we can apply the predictions

newData, newSR = librosa.load("../input/turkish-speech-dataset/normal2/export (46).wav"

                              ,duration=2.5

                              ,sr=44100

                              ,offset=0)



newSR = np.array(newSR)

mfccs = np.mean(librosa.feature.mfcc(y=newData, sr=newSR, n_mfcc=13),axis=0)

newdf = pd.DataFrame(data=mfccs).T

print(newdf.shape)



newdf= np.expand_dims(newdf,axis=2)

print(newdf.shape)

newpred=model.predict(newdf)



filename = filename = './labels'

infile = open(filename,'rb')

lb = pickle.load(infile)

infile.close()



# Get the final predicted label

final = newpred.argmax(axis=1)

final = final.astype(int).flatten()

final = (lb.inverse_transform((final)))

print(final) #emo(final) #gender(final) 
newData,newSR= librosa.load("../input/duygu-test/lanet olsun dostum.wav")

ipd.Audio("../input/duygu-test/lanet olsun dostum.wav")
plt.figure(figsize=(15, 5))

librosa.display.waveplot(newData, sr=newSR)



# Lets transform the dataset so we can apply the predictions

newData, newSR = librosa.load("../input/duygu-test/lanet olsun dostum.wav"

                              ,duration=2.5

                              ,sr=44100

                              ,offset=0)



newSR = np.array(newSR)

mfccs = np.mean(librosa.feature.mfcc(y=newData, sr=newSR, n_mfcc=13),axis=0)

newdf = pd.DataFrame(data=mfccs).T

print(newdf.shape)



newdf= np.expand_dims(newdf,axis=2)

print(newdf.shape)

newpred=model.predict(newdf)



filename = filename = './labels'

infile = open(filename,'rb')

lb = pickle.load(infile)

infile.close()



# Get the final predicted label

final = newpred.argmax(axis=1)

final = final.astype(int).flatten()

final = (lb.inverse_transform((final)))

print(final) #emo(final) #gender(final) 