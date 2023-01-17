# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
np.random.seed(1001)
import librosa
import scipy
import os
import shutil #working with files and collections
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook  # means “progress” 
from sklearn.cross_validation import StratifiedKFold
import wave
# Using scipy
from scipy.io import wavfile
import keras # neural networks
import tensorflow as tf # neural networks
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (Convolution2D, Dense, Dropout, GlobalAveragePooling2D, 
                              GlobalMaxPool2D, Input, MaxPool2D, concatenate, Activation,  
                              MaxPooling2D,Flatten,BatchNormalization, Conv2D,AveragePooling2D)
from keras.utils import Sequence, to_categorical
import math
from sklearn.model_selection import StratifiedShuffleSplit

# https://github.com/titu1994/Snapshot-Ensembles
from snapshot import SnapshotCallbackBuilder

import os

matplotlib.style.use('ggplot')

# Any results you write to the current directory are saved as output.
train = pd.read_csv("D:/urban-sound-classification/train/train.csv",index_col=None)
print("Number of training examples=", train.shape[0], "  Number of classes=", len(train.Class.unique()))
test = os.listdir("D:/urban-sound-classification/test/Test")
len(test)
duration_test = []
duration_train = []

train.head(10)

#############   Duration
non4s = []
for idd,i in enumerate(train["ID"]):
    fname = 'D:/urban-sound-classification/train/train/' + str(i) + ".wav"
    y, sr = librosa.load(fname,sr=None)
    dur = librosa.get_duration(y=y, sr=sr)
    if dur != 4.0:
        non4s.append(idd)
    duration_train.append(dur)
print("non4s = ", len(non4s))    
pd.DataFrame(duration_train).describe()
train = train.drop(non4s)



print(train.head(10))
split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in split.split(train,train["label"]):
    strat_train = train.loc[train_index]
    strat_test = train.loc[test_index]
strat_train.head()
strat_test.head()

strat_train = strat_train.dropna()
strat_test = strat_test.dropna()


print(strat_train.groupby(['label']).count())
print(strat_test.groupby(['label']).count())

fname = 'D:/urban-sound-classification/train/train/' + '6' + ".wav"
y, sr = librosa.load(fname,sr=22400)
#y_res = librosa.core.resample(y, sr, sr, res_type='kaiser_best', fix=True, scale=True)
bandwidth = librosa.feature.spectral_bandwidth(y, sr)
centr = librosa.feature.spectral_centroid(y,sr)
rmse = librosa.feature.rmse(y=y)
spectral_contrast = librosa.feature.spectral_contrast(y,sr)
tonnetz = librosa.feature.tonnetz(y,sr)
zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
tempogram = librosa.feature.tempogram(y,sr)

class Config(object):
    def __init__(self,
                 sampling_rate=8000, audio_duration=2, n_classes=9,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001, 
                 max_epochs=50, n_mfcc=20):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        #self.audio_length = self.sampling_rate * self.audio_duration
        self.audio_length = 4
#         if self.use_mfcc:
#             self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
#         else:
#             self.dim = (self.audio_length, 1)
        #self.dim = (16, 77, 1)
        self.dim = (16, 11, 1)
        #self.dim = (24, 44, 1)
        #self.dim = (384, 176, 1)

def prepare_data(df, config):
    X = np.empty(shape=(df.shape[0], 40))
    input_length = config.audio_length
    for i, fname in enumerate(df):
        #print(fname)
        file_path = 'D:/urban-sound-classification/train/train/' + str(int(fname)) + ".wav"
        data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")
        data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
        #data = librosa.feature.delta(data)
        data = np.apply_along_axis(np.mean,1,data)
        data = data.reshape(1,40)
        #data = np.expand_dims(data, axis=-1)
        X[i,] = data 
        
    return X

split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in split.split(train,train["label"]):
    strat_train = train.loc[train_index]
    strat_test = train.loc[test_index]
strat_train.head()
strat_test.head()

strat_train = strat_train.dropna()
strat_test = strat_test.dropna()


print(strat_train.groupby(['label']).count())
print(strat_test.groupby(['label']).count())

### generate MFCC train data
class Config(object):
    def __init__(self,
                 sampling_rate=8000, audio_duration=2, n_classes=9,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001, 
                 max_epochs=50, n_mfcc=20):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        #self.audio_length = self.sampling_rate * self.audio_duration
        self.audio_length = 4
#         if self.use_mfcc:
#             self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
#         else:
#             self.dim = (self.audio_length, 1)
        self.dim = (5, 8, 1)

def prepare_data(df, config):
    X = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1], 1))
    input_length = config.audio_length
    for i, fname in enumerate(df):
        #print(fname)
        file_path = 'D:/urban-sound-classification/train/train/' + str(int(fname)) + ".wav"
        data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")

#         # Random offset / Padding
#         if len(data) > input_length:
#             max_offset = len(data) - input_length
#             offset = np.random.randint(max_offset)
#             data = data[offset:(input_length+offset)]
#         else:
#             if input_length > len(data):
#                 max_offset = input_length - len(data)
#                 offset = np.random.randint(max_offset)
#             else:
#                 offset = 0
#             data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

        data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
        data = np.apply_along_axis(np.mean,1,data)
        data = data.reshape(config.dim[0],config.dim[1])
        data = np.expand_dims(data, axis=-1)
        X[i,] = data
    return X


def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5


config = Config(sampling_rate=50000, audio_duration=4, n_folds=10, 
                learning_rate=0.001, use_mfcc=True, n_mfcc=40)  

X_mfcc_train=prepare_data(strat_train["fname"], config)
y_train = strat_train["label"]
X_mfcc_test=prepare_data(strat_test["fname"], config)
y_test = strat_test["label"]

print(y_train[:10])
print(strat_train[:10])
print(len(pd.unique(y_train)))
print(len(pd.unique(y_test)))

y_train = to_categorical(y_train, num_classes=config.n_classes)
y_test = to_categorical(y_test, num_classes=config.n_classes)

X_mfcc_train = X_mfcc_train.reshape(X_mfcc_train.shape[0],data_rows, data_cols,1)
X_mfcc_test = X_mfcc_test.reshape(X_mfcc_test.shape[0],data_rows, data_cols,1)
input_shape = (data_rows, data_cols, 1)


kernel_initializer='lecun_uniform'
bias_initializer='zeros'
kernel_regularizer=None
activation = "selu"


############ Визначення шарів згорткової нейронної мережі
model = models.Sequential()
model.add(Conv2D(128, 32, 32, border_mode="same", 
                 input_shape = input_shape,kernel_initializer=kernel_initializer, 
                 bias_initializer=bias_initializer, kernel_regularizer=None))
model.add(BatchNormalization())
model.add(Activation(activation))
model.add(AveragePooling2D())
############ Додавання повнозв'язного шару 
model.add(Flatten())
model.add(Dense(1024,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer))
model.add(Activation("relu"))
model.add(Dropout(0.6))
model.add(Dense(1024,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer))
model.add(Activation("relu"))
model.add(Dropout(0.8))
model.add(Dense(9,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer))
model.add(Activation('softmax'))
############ Компіляція моделі
############ Головні параметри ансамблю
M = 150 # Кількість станів, які зберігаються
nb_epoch = T = 200 # Кількість епох навчання
alpha_zero = 0.0001 # Коефіцієнт швидкості навчання
model_prefix = 'Model_'
snapshot = SnapshotCallbackBuilder(T, M, alpha_zero) 
optimizer = optimizers.Nadam(lr=alpha_zero, beta_1=0.9, beta_2=0.999, 
                             epsilon=None, schedule_decay=0.004)
model.compile(loss = "categorical_crossentropy", optimizer = optimizer, 
              metrics = ["accuracy"])
history = model.fit(X_mfcc_train, y_train, batch_size = batch_size, 
                    epochs = nb_epoch, verbose=2, validation_data = (X_mfcc_test, y_test),
                    callbacks=snapshot.get_callbacks(model_prefix=model_prefix))
score = model.evaluate(X_mfcc_test, y_test,verbose = 0)
print("test score: %f" % score[0])
print("test accuracy: %f" % score[1])
model.summary()

#model1 = AdaBoostClassifier(base_estimator=model, n_estimators=50000, learning_rate=0.0001, algorithm='SAMME.R', random_state=None).fit(X_mfcc_train,test)

from keras.utils import plot_model
plot_model(model, to_file='model.png')

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'], 'k-')
plt.plot(history.history['val_acc'], 'k:')
plt.title('Точність моделі')
plt.ylabel('точність')
plt.xlabel('епохи')
plt.legend(['тренування', 'тест'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'], 'k-')
plt.plot(history.history['val_loss'], 'k:')
plt.title('Похибка моделі')
plt.ylabel('похибка')
plt.xlabel('епохи')
plt.legend(['тренування', 'тест'], loc='upper left')
plt.show()

proba = model.predict_classes(X_mfcc_test)
from sklearn.metrics import confusion_matrix
print(proba)
print(test)
cm = confusion_matrix(test,proba)
print('confusion_matrix\n',cm)

plt.matshow(cm,cmap=plt.cm.gray)
plt.show()