# Importing required libraries 

# Keras

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



# sklearn

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



# Other  

import librosa

import librosa.display

import json

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from matplotlib.pyplot import specgram

import pandas as pd

import seaborn as sns

import glob 

import os

from tqdm import tqdm

import pickle

import IPython.display as ipd  # To play sound in the notebook
#########################

# Augmentation methods

#########################

def noise(data):

    """

    Adding White Noise.

    """

    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html

    noise_amp = 0.05*np.random.uniform()*np.amax(data)   # more noise reduce the value to 0.5

    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])

    return data

    

def shift(data):

    """

    Random Shifting.

    """

    s_range = int(np.random.uniform(low=-5, high = 5)*1000)  #default at 500

    return np.roll(data, s_range)

    

def stretch(data, rate=0.8):

    """

    Streching the Sound. Note that this expands the dataset slightly

    """

    data = librosa.effects.time_stretch(data, rate)

    return data

    

def pitch(data, sample_rate):

    """

    Pitch Tuning.

    """

    bins_per_octave = 12

    pitch_pm = 2

    pitch_change =  pitch_pm * 2*(np.random.uniform())   

    data = librosa.effects.pitch_shift(data.astype('float64'), 

                                      sample_rate, n_steps=pitch_change, 

                                      bins_per_octave=bins_per_octave)

    return data

    

def dyn_change(data):

    """

    Random Value Change.

    """

    dyn_change = np.random.uniform(low=-0.5 ,high=7)  # default low = 1.5, high = 3

    return (data * dyn_change)

    

def speedNpitch(data):

    """

    peed and Pitch Tuning.

    """

    # you can change low and high here

    length_change = np.random.uniform(low=0.8, high = 1)

    speed_fac = 1.2  / length_change # try changing 1.0 to 2.0 ... =D

    tmp = np.interp(np.arange(0,len(data),speed_fac),np.arange(0,len(data)),data)

    minlen = min(data.shape[0], tmp.shape[0])

    data *= 0

    data[0:minlen] = tmp[0:minlen]

    return data



####################################

# the confusion matrix heat map plot

####################################

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    

    Arguments

    ---------

    confusion_matrix: numpy.ndarray

        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 

        Similarly constructed ndarrays can also be used.

    class_names: list

        An ordered list of class names, in the order they index the given confusion matrix.

    figsize: tuple

        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,

        the second determining the vertical size. Defaults to (10,7).

    fontsize: int

        Font size for axes labels. Defaults to 14.

        

    Returns

    -------

    matplotlib.figure.Figure

        The resulting confusion matrix figure

    """

    df_cm = pd.DataFrame(

        confusion_matrix, index=class_names, columns=class_names, 

    )

    fig = plt.figure(figsize=figsize)

    try:

        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    except ValueError:

        raise ValueError("Confusion matrix values must be integers.")

        

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
# Use one audio file in previous parts again

fname = '/kaggle/input/surrey-audiovisual-expressed-emotion-savee/ALL/JK_f11.wav'  

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Paly it again to refresh our memory

ipd.Audio(data, rate=sampling_rate)
x = noise(data)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(x, sr=sampling_rate)

ipd.Audio(x, rate=sampling_rate)
x = shift(data)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(x, sr=sampling_rate)

ipd.Audio(x, rate=sampling_rate)
x = stretch(data)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(x, sr=sampling_rate)

ipd.Audio(x, rate=sampling_rate)
x = pitch(data, sampling_rate)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(x, sr=sampling_rate)

ipd.Audio(x, rate=sampling_rate)
x = dyn_change(data)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(x, sr=sampling_rate)

ipd.Audio(x, rate=sampling_rate)
x = speedNpitch(data)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(x, sr=sampling_rate)

ipd.Audio(x, rate=sampling_rate)
# lets pick up the meta-data that we got from our first part of the Kernel

ref = pd.read_csv("/kaggle/input/data-path/Data_path.csv")

ref.head()
# Note this takes a couple of minutes (~16 mins) as we're iterating over 4 datasets, and with augmentation  

df = pd.DataFrame(columns=['feature'])

df_noise = pd.DataFrame(columns=['feature'])

df_speedpitch = pd.DataFrame(columns=['feature'])

cnt = 0



# loop feature extraction over the entire dataset

for i in tqdm(ref.path):

    

    # first load the audio 

    X, sample_rate = librosa.load(i

                                  , res_type='kaiser_fast'

                                  ,duration=2.5

                                  ,sr=44100

                                  ,offset=0.5

                                 )



    # take mfcc and mean as the feature. Could do min and max etc as well. 

    mfccs = np.mean(librosa.feature.mfcc(y=X, 

                                        sr=np.array(sample_rate), 

                                        n_mfcc=13),

                    axis=0)

    

    df.loc[cnt] = [mfccs]   



    # random shifting (omit for now)

    # Stretch

    # pitch (omit for now)

    # dyn change

    

    # noise 

    aug = noise(X)

    aug = np.mean(librosa.feature.mfcc(y=aug, 

                                    sr=np.array(sample_rate), 

                                    n_mfcc=13),    

                  axis=0)

    df_noise.loc[cnt] = [aug]



    # speed pitch

    aug = speedNpitch(X)

    aug = np.mean(librosa.feature.mfcc(y=aug, 

                                    sr=np.array(sample_rate), 

                                    n_mfcc=13),    

                  axis=0)

    df_speedpitch.loc[cnt] = [aug]   



    cnt += 1



df.head()
# combine 

df = pd.concat([ref,pd.DataFrame(df['feature'].values.tolist())],axis=1)

df_noise = pd.concat([ref,pd.DataFrame(df_noise['feature'].values.tolist())],axis=1)

df_speedpitch = pd.concat([ref,pd.DataFrame(df_speedpitch['feature'].values.tolist())],axis=1)

print(df.shape,df_noise.shape,df_speedpitch.shape)
df = pd.concat([df,df_noise,df_speedpitch],axis=0,sort=False)

df=df.fillna(0)

del df_noise, df_speedpitch



df.head()
# Split between train and test 

X_train, X_test, y_train, y_test = train_test_split(df.drop(['path','labels','source'],axis=1)

                                                    , df.labels

                                                    , test_size=0.25

                                                    , shuffle=True

                                                    , random_state=42

                                                   )



# Lets see how the data present itself before normalisation 

X_train[150:160]
# Lts do data normalization 

mean = np.mean(X_train, axis=0)

std = np.std(X_train, axis=0)



X_train = (X_train - mean)/std

X_test = (X_test - mean)/std



# Check the dataset now 

X_train[150:160]
# Lets few preparation steps to get it into the correct format for Keras 

X_train = np.array(X_train)

y_train = np.array(y_train)

X_test = np.array(X_test)

y_test = np.array(y_test)



# one hot encode the target 

lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))

y_test = np_utils.to_categorical(lb.fit_transform(y_test))



print(X_train.shape)

print(lb.classes_)



# Pickel the lb object for future use 

filename = 'labels'

outfile = open(filename,'wb')

pickle.dump(lb,outfile)

outfile.close()
X_train = np.expand_dims(X_train, axis=2)

X_test = np.expand_dims(X_test, axis=2)

X_train.shape
# New model

model = Sequential()

model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1)))  # X_train.shape[1] = No. of Columns

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

model.add(Dense(14)) # Target class number

model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

model_history=model.fit(X_train, y_train, batch_size=16, epochs=150, validation_data=(X_test, y_test),verbose=2)
plt.plot(model_history.history['loss'])

plt.plot(model_history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# Save model and weights

model_name = 'Emotion_Model_aug.h5'

save_dir = os.path.join(os.getcwd(), 'saved_models')



if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)

model.save(model_path)

print('Save model and weights at %s ' % model_path)



# Save the model to disk

model_json = model.to_json()

with open("model_json_aug.json", "w") as json_file:

    json_file.write(model_json)
# loading json and model architecture 

json_file = open('model_json_aug.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



# load weights into new model

loaded_model.load_weights("saved_models/Emotion_Model_aug.h5")

print("Loaded model from disk")

 

# Keras optimiser

opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

score = loaded_model.evaluate(X_test, y_test, verbose=0)

print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
preds = loaded_model.predict(X_test, 

                         batch_size=16, 

                         verbose=1)



preds=preds.argmax(axis=1)

preds
# predictions 

preds = preds.astype(int).flatten()

preds = (lb.inverse_transform((preds)))

preds = pd.DataFrame({'predictedvalues': preds})



# Actual labels

actual=y_test.argmax(axis=1)

actual = actual.astype(int).flatten()

actual = (lb.inverse_transform((actual)))

actual = pd.DataFrame({'actualvalues': actual})



# Lets combined both of them into a single dataframe

finaldf = actual.join(preds)

finaldf[170:180]
# Write out the predictions to disk

finaldf.to_csv('Predictions.csv', index=False)

finaldf.groupby('predictedvalues').count()
# Get the predictions file 

finaldf = pd.read_csv("Predictions.csv")

classes = finaldf.actualvalues.unique()

classes.sort()    



# Confusion matrix 

c = confusion_matrix(finaldf.actualvalues, finaldf.predictedvalues)

print(accuracy_score(finaldf.actualvalues, finaldf.predictedvalues))

print_confusion_matrix(c, class_names = classes)
# Classification report 

classes = finaldf.actualvalues.unique()

classes.sort()    

print(classification_report(finaldf.actualvalues, finaldf.predictedvalues, target_names=classes))
modidf = finaldf

modidf['actualvalues'] = finaldf.actualvalues.replace({'female_angry':'female'

                                       , 'female_disgust':'female'

                                       , 'female_fear':'female'

                                       , 'female_happy':'female'

                                       , 'female_sad':'female'

                                       , 'female_surprise':'female'

                                       , 'female_neutral':'female'

                                       , 'male_angry':'male'

                                       , 'male_fear':'male'

                                       , 'male_happy':'male'

                                       , 'male_sad':'male'

                                       , 'male_surprise':'male'

                                       , 'male_neutral':'male'

                                       , 'male_disgust':'male'

                                      })



modidf['predictedvalues'] = finaldf.predictedvalues.replace({'female_angry':'female'

                                       , 'female_disgust':'female'

                                       , 'female_fear':'female'

                                       , 'female_happy':'female'

                                       , 'female_sad':'female'

                                       , 'female_surprise':'female'

                                       , 'female_neutral':'female'

                                       , 'male_angry':'male'

                                       , 'male_fear':'male'

                                       , 'male_happy':'male'

                                       , 'male_sad':'male'

                                       , 'male_surprise':'male'

                                       , 'male_neutral':'male'

                                       , 'male_disgust':'male'

                                      })



classes = modidf.actualvalues.unique()  

classes.sort() 



# Confusion matrix 

c = confusion_matrix(modidf.actualvalues, modidf.predictedvalues)

print(accuracy_score(modidf.actualvalues, modidf.predictedvalues))

print_confusion_matrix(c, class_names = classes)
# Classification report 

classes = modidf.actualvalues.unique()

classes.sort()    

print(classification_report(modidf.actualvalues, modidf.predictedvalues, target_names=classes))
modidf = pd.read_csv("Predictions.csv")

modidf['actualvalues'] = modidf.actualvalues.replace({'female_angry':'angry'

                                       , 'female_disgust':'disgust'

                                       , 'female_fear':'fear'

                                       , 'female_happy':'happy'

                                       , 'female_sad':'sad'

                                       , 'female_surprise':'surprise'

                                       , 'female_neutral':'neutral'

                                       , 'male_angry':'angry'

                                       , 'male_fear':'fear'

                                       , 'male_happy':'happy'

                                       , 'male_sad':'sad'

                                       , 'male_surprise':'surprise'

                                       , 'male_neutral':'neutral'

                                       , 'male_disgust':'disgust'

                                      })



modidf['predictedvalues'] = modidf.predictedvalues.replace({'female_angry':'angry'

                                       , 'female_disgust':'disgust'

                                       , 'female_fear':'fear'

                                       , 'female_happy':'happy'

                                       , 'female_sad':'sad'

                                       , 'female_surprise':'surprise'

                                       , 'female_neutral':'neutral'

                                       , 'male_angry':'angry'

                                       , 'male_fear':'fear'

                                       , 'male_happy':'happy'

                                       , 'male_sad':'sad'

                                       , 'male_surprise':'surprise'

                                       , 'male_neutral':'neutral'

                                       , 'male_disgust':'disgust'

                                      })



classes = modidf.actualvalues.unique() 

classes.sort() 



# Confusion matrix 

c = confusion_matrix(modidf.actualvalues, modidf.predictedvalues)

print(accuracy_score(modidf.actualvalues, modidf.predictedvalues))

print_confusion_matrix(c, class_names = classes)
# Classification report 

classes = modidf.actualvalues.unique()

classes.sort()    

print(classification_report(modidf.actualvalues, modidf.predictedvalues, target_names=classes))