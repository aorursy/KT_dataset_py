# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
%matplotlib inline
import os,random
os.environ["KERAS_BACKEND"] = "theano"
#os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["THEANO_FLAGS"]  = "device=cuda%d"%(1)
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, random, sys, keras
# Load the dataset ...
#  You will need to seperately download or generate this file
with open("/kaggle/input/RML2016.10a_dict.pkl",'rb') as file:
    Xd = pickle.load(file,encoding='bytes')
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)
#  into training and test sets of the form we can train/test on 
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples // 2
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D , Reshape , ZeroPadding2D,BatchNormalization,LSTM
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
dr = 0.5 # dropout rate (%)
model.add(LSTM(128, input_shape=X_train.shape[1:], return_sequences=True))
model.add(Reshape([1]+in_shp))#, input_shape=(2,200)))
model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(256, 1, 3, padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform',data_format='channels_first'))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(80, 2, 3, padding='valid', activation="relu", name="conv2", kernel_initializer='glorot_uniform',data_format='channels_first'))
# model.add(LSTM(2240, input_shape=X_train.shape[1:], return_sequences=True))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense( len(classes), kernel_initializer='he_normal', name="dense2" ))
# model.add(LSTM((11, ), input_shape=X_train.shape))
model.add(Activation('softmax'))

model.add(Reshape([len(classes)]))
# model.add(LSTM(11, input_shape=X_train.shape[1:]))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
model.summary()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D , Reshape , ZeroPadding2D,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
# Set up some params 
nb_epoch = 200     # number of epochs to train on
batch_size = 1024  # training batch size
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=3, verbose=1, factor=0.5, 
                                            min_lr=0.000001, cooldown=2)
# perform training ...
#   - call the main training loop in keras for our network+dataset
filepath = 'convmodrecnets_CNN2_0.5.wts.h5'
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    nb_epoch=nb_epoch,
   # show_accuracy=False,
    verbose=2,
    validation_data=(X_test, Y_test),
    callbacks = [
#                  learning_rate_reduction,
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto',baseline=None, restore_best_weights=False)
    ])
# we re-load the best weights once training is finished
model.load_weights(filepath)
# Show simple version of performance
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print (score)
# Show loss curves 
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Plot confusion matrix
test_Y_hat = model.predict(X_test, batch_size=batch_size)
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = list(Y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, labels=classes)
     # Plot confusion matrix
acc = {}
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
            j = list(test_Y_i[i,:]).index(1)
            k = int(np.argmax(test_Y_i_hat[i,:]))
            conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
            confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d) using LSTM"%(snr))
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor

    print("snr",snr)
    print ("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)
print(acc)
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN Classification Accuracy on dataset RadioML 2016.10 Alpha")
CC= Xd.keys()
dd=list(CC)
dd
print(Xd.keys())
def modulation1(str12):
            plt.plot(Xd[str12,4][2,0])
            plt.plot(Xd[str12,8][2,0])
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title("%s Time Plot"%(str12))
            plt.grid(b=True, axis='both')
str2=b'QAM64'
modulation1(str2)

str2=b'QPSK'
modulation1(str2)
print(mods)
str2=b'8PSK'
modulation1(str2)
str2=b'AM-DSB'
modulation1(str2)
str2=b'AM-SSB'
modulation1(str2)
str2=b'BPSK'
modulation1(str2)
str2=b'CPFSK'
modulation1(str2)
str2=b'GFSK'
modulation1(str2)
str2=b'PAM4'
modulation1(str2)
str2=b'QAM16'
modulation1(str2)
str2=b'WBFM'
modulation1(str2)
import matplotlib.pyplot as plt

def my_function(str1):
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    X1 = []
    Y1=[]
    lbl = []
    for mod in mods:
       # print(mod)
        for snr in snrs:
            if(mod==str1 and snr==10):
                    #print("test1")
                test = Xd[(mod,snr)]
                    #print(test)
                X1.append(Xd[(mod,snr)])
                for i in range(Xd[(mod,snr)].shape[0]):  
                    lbl.append((mod,snr))
                    Y1.append([test[0]+1j*test[1]])
    X1 = np.vstack(X1)
    Y1 = np.vstack(Y1)

    df= pd.DataFrame(lbl,columns=["mod","snr"])
    df['snr'].value_counts()
    ind = []
    for i in range(0,df.shape[0]):
        if(df['snr'][i]==10):
            ind.append(i)
    #print(len(ind))

    for i in range(0,100,1):
            x = X1[i][0]
            y= X1[i][1]
            fig = plt.figure()
            plt.scatter(x,y,c='blue',label=i)
            plt.xlabel("I")
            plt.ylabel("Q")
            plt.title("Data representation variance in %s SNR 10" %(str1) )
            plt.legend()
print(mods)
#str1=mods[1]
#print(str1)
#print(str)
for str1 in mods:
    #str1=mods[j]
    my_function(str1)
