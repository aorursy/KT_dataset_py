# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import librosa
import keras
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Sequential
from keras.layers import MaxPooling2D,Conv2D,Flatten,Activation,Dense,Dropout,BatchNormalization
from keras.optimizers import Adamax
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os  # Manipulate files
from  matplotlib import pyplot as plt
from IPython.display import clear_output
# Any results you write to the current directory are saved as output.
# List the wav files
ROOT_DIR = '../input/cats_dogs/'
X_path = os.listdir(ROOT_DIR)
print (X_path)
# changing the values into 1 and 0
y = [0 if 'cat' in f else 1 for f in X_path]  # change y to int values
print (y)
# Split train and test
train_input, test_input, train_target, test_target = train_test_split(X_path, y, test_size=0.10)
def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name,duration=5)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz
i = 0
X_train = []
while i<len(train_input):
    print('processing file: ',i)
    filename = '../input/cats_dogs/' + train_input[i]
    mfccs, chroma, mel, contrast,tonnetz = extract_feature(filename)
    features = []
    features.append(np.mean(mfccs))
    features.append(np.mean(chroma))
    features.append(np.mean(mel))
    features.append(np.mean(contrast))
    features.append(np.mean(tonnetz))
    X_train.append(features)
    i = i +1
# converting into an numpy array
X_train = np.asarray(X_train)
y_train = np.asarray(train_target)
print(X_train.shape)
print(y_train.shape)
model = Sequential()
model.add(Dense(500,input_shape = (5,)))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()
model.compile(optimizer = 'adam', metrics=['accuracy'], loss = 'binary_crossentropy')
#plot
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []
        
    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()
        
plot_losses = PlotLosses()

#training the model
model.fit(X_train,y_train,epochs=500,callbacks = [plot_losses],verbose= 2)
i = 0
X_test = []
while i<len(test_input):
   print('processing file: ',i)
   filename = '../input/cats_dogs/' + test_input[i]
   mfccs, chroma, mel, contrast,tonnetz = extract_feature(filename)
   features = []
   features.append(np.mean(mfccs))
   features.append(np.mean(chroma))
   features.append(np.mean(mel))
   features.append(np.mean(contrast))
   features.append(np.mean(tonnetz))
   X_test.append(features)
   i = i +1
X_test = np.asarray(X_test)
predicted = model.predict(X_test)
#changing the output as 0 or 1
i = 0
while i<len(predicted):
   if predicted[i] >=.5:
      predicted[i] = 1
   else:
      predicted[i] = 0
   i = i +1
y_test = np.asarray(predicted)
predicted = predicted.reshape([-1])
print(predicted.shape)
#plotting the confusion matrix
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
       
        #This function prints and plots the confusion matrix.
        #Normalization can be applied by setting `normalize=True`.
        
        if normalize:
           cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
           print("Normalized confusion matrix")
        else:
           print('Confusion matrix, without normalization')
        
        print(cm)
        
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
           plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predicted)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['cat','dog'],title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['cat','dog'], normalize=True,title='Normalized confusion matrix')

plt.show()
