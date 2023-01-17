# Other Libraries 

import librosa

import librosa.display

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from matplotlib.pyplot import specgram

import pandas as pd

import glob 

from sklearn.metrics import confusion_matrix

import IPython.display as ipd  # To play sound in the notebook

import os

import sys

import warnings

import pickle

import seaborn as sns



# Keras

import keras

from keras import regularizers

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, Model, model_from_json

from keras.layers import Dense, Embedding, LSTM

from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization

from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, LeakyReLU

from keras.utils import np_utils, to_categorical

from keras.callbacks import ModelCheckpoint



# sklearn

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
TESS = TESS = "/kaggle/input/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"
#TESS Dataset



path = []

emotion = []

dir_list = os.listdir(TESS)

dir_list.sort()

for i in dir_list:

    fname = os.listdir(TESS+i)

    for f in fname:

        if i == 'OAF_angry' or i == 'YAF_angry':

            emotion.append('angry')

        elif i == 'OAF_disgust' or i == 'YAF_disgust':

            emotion.append('disgust')

        elif i == 'OAF_Fear' or i == 'YAF_fear':

            emotion.append('fear')

        elif i == 'OAF_happy' or i == 'YAF_happy':

            emotion.append('happy')

        elif i == 'OAF_neutral' or i == 'YAF_neutral':

            emotion.append('neutral')                                

        elif i == 'OAF_Pleasant_surprise' or i == 'YAF_pleasant_surprised':

            emotion.append('surprise')               

        elif i == 'OAF_Sad' or i == 'YAF_sad':

            emotion.append('sad')

        else:

            emotion.append('Unknown')

        path.append(TESS + i + "/" + f)



TESS_df = pd.DataFrame(emotion, columns = ['labels'])

TESS_df['source'] = 'TESS'

TESS_df = pd.concat([TESS_df,pd.DataFrame(path, columns = ['path'])],axis=1)

TESS_df.labels.value_counts()
ref = pd.concat([TESS_df], axis = 0)

print(ref.labels.value_counts())

ref.head()
# Note this takes a couple of minutes (~10 mins) as we're iterating over 4 datasets 

df = pd.DataFrame(columns=['feature'])



# loop feature extraction over the entire dataset

counter=0

for index,path in enumerate(ref.path):

    X, sample_rate = librosa.load(path

                                  , res_type='kaiser_fast'

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
mfcc_df = pd.DataFrame(df['feature'].values.tolist())

ref = ref.reset_index()

ref = ref.drop(columns = 'index') 

df = pd.concat([ref,mfcc_df],axis=1)
# replace NA with 0

df=df.fillna(0)
# Split between train and test 

X_train, X_test, y_train, y_test = train_test_split(df.drop(['path','labels','source'],axis=1)

                                                    , df.labels

                                                    , test_size=0.25

                                                    , shuffle=True

                                                    , random_state=42

                                                   )
# Lts do data normalization 

mean = np.mean(X_train, axis=0)

std = np.std(X_train, axis=0)



X_train = (X_train - mean)/std

X_test = (X_test - mean)/std
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
# the model below gives 87% accuracy 



model = Sequential()

model.add(Dense(149, input_shape = (X_train.shape[1], ), activation ='relu', use_bias=True))

model.add(Dense(7, activation='softmax', use_bias=True))

opt = keras.optimizers.Adam(lr=0.0005)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

model_history=model.fit(X_train, y_train, batch_size=8, epochs=14, validation_data=(X_test, y_test))



plt.plot(model_history.history['loss'])

plt.plot(model_history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



preds = model.predict(X_test, 

                         batch_size=16, 

                         verbose=1)



preds=preds.argmax(axis=1)



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



# Classification report 

classes = finaldf.actualvalues.unique()

classes.sort()    

print(classification_report(finaldf.actualvalues, finaldf.predictedvalues, target_names=classes))
X_train = np.expand_dims(X_train, axis=2)

X_test = np.expand_dims(X_test, axis=2)

X_train.shape
model = Sequential()

model.add(Conv1D(filters = 10, kernel_size = 100 , strides = 1, padding='same',use_bias = True, input_shape=(X_train.shape[1],1))) 

model.add(MaxPooling1D(pool_size=(8)))

model.add(Flatten())

model.add(Dense(400, activation ='relu', use_bias=True))

model.add(Dense(150, activation ='relu', use_bias=True))

model.add(Dense(7, activation='softmax', use_bias=True))

opt = keras.optimizers.Adam(lr=0.0001)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

model_history=model.fit(X_train, y_train, batch_size=16, epochs=25, validation_data=(X_test, y_test))



plt.plot(model_history.history['loss'])

plt.plot(model_history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

preds = model.predict(X_test, 

                         batch_size=16, 

                         verbose=1)



preds=preds.argmax(axis=1)



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



# Classification report 

classes = finaldf.actualvalues.unique()

classes.sort()    

print(classification_report(finaldf.actualvalues, finaldf.predictedvalues, target_names=classes))
# the confusion matrix heat map plot

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

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

# Confusion matrix 

c = confusion_matrix(finaldf.actualvalues, finaldf.predictedvalues)

print_confusion_matrix(c, class_names = classes)