# Importing required libraries 

from keras.models import Sequential, Model, model_from_json

import matplotlib.pyplot as plt

import keras 

import pickle

import wave  # !pip install wave

import os

import pandas as pd

import numpy as np

import sys

import warnings

import librosa

import librosa.display

import IPython.display as ipd  # To play sound in the notebook



# ignore warnings 

if not sys.warnoptions:

    warnings.simplefilter("ignore")
data, sampling_rate = librosa.load('/kaggle/input/happy-audio/Liza-happy-v3.wav')

ipd.Audio('/kaggle/input/happy-audio/Liza-happy-v3.wav')
plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)
# loading json and model architecture 

json_file = open('/kaggle/input/saved-model/model_json.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



# load weights into new model

loaded_model.load_weights("/kaggle/input/saved-model/Emotion_Model.h5")

print("Loaded model from disk")



# the optimiser

opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# Lets transform the dataset so we can apply the predictions

X, sample_rate = librosa.load('/kaggle/input/happy-audio/Liza-happy-v3.wav'

                              ,res_type='kaiser_fast'

                              ,duration=2.5

                              ,sr=44100

                              ,offset=0.5

                             )



sample_rate = np.array(sample_rate)

mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)

newdf = pd.DataFrame(data=mfccs).T

newdf
# Apply predictions

newdf= np.expand_dims(newdf, axis=2)

newpred = loaded_model.predict(newdf, 

                         batch_size=16, 

                         verbose=1)



newpred
filename = '/kaggle/input/labels/labels'

infile = open(filename,'rb')

lb = pickle.load(infile)

infile.close()



# Get the final predicted label

final = newpred.argmax(axis=1)

final = final.astype(int).flatten()

final = (lb.inverse_transform((final)))

print(final) #emo(final) #gender(final) 