# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math # para ceil() en particular



import h5py

import matplotlib.pyplot as plt

from keras import optimizers

from keras import backend as K

from keras.datasets import mnist

from keras.utils import np_utils, to_categorical

from keras.layers import Dense, Dropout, Flatten, LeakyReLU, Activation

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras.regularizers import l2

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split



%matplotlib inline

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'



%load_ext autoreload

%autoreload 2



np.random.seed(1)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
K.set_image_data_format('channels_last')
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
train_dfX = train_df.drop(['label'], axis=1)

train_dfY = train_df['label'] 

# Utilizando split de 80% test - 20% validation

# También se usa .values para obtener numpy arrays en vez de dataframes

train_X, val_X, train_Y, val_Y = train_test_split(train_dfX.values, train_dfY.values, test_size=0.2,stratify=train_dfY)

print("Entrenamiento: ",train_X.shape)

print("Validacion:    ",val_X.shape)
train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)

val_X = val_X.reshape(val_X.shape[0], 28, 28, 1)
unique, count= np.unique(train_Y, return_counts=True)

print("Distribucion de los labels en el train set: %s " % dict (zip(unique, np.round(count/train_Y.shape[0],3)) ), "\n" )



unique, count= np.unique(val_Y, return_counts=True)

print("Distribucion de los labels en el validation set: %s " % dict (zip(unique, np.round(count/val_Y.shape[0],3)) ), "\n" )
labels_indices = np.unique(train_Y,return_index=True)

for label in labels_indices[0]:

    plt.subplot(2, 5, label + 1)

    plt.axis('off')

    plt.imshow(train_X[labels_indices[1][label]].squeeze(), cmap=plt.cm.gray_r, interpolation='nearest')

    plt.title('label: %i' % label )
NUM_CLASSES = labels_indices[0].size
datagen = ImageDataGenerator(

    rescale = 1/255, # Solo nos interesa valores entre 0 y 1

    rotation_range = 20, # Rango de rotación de 20°

    zoom_range = 0.2 # Rango del zoom

)

train_X_Aug = np.array(train_X, copy=True)

train_Y_Aug = np.array(train_Y, copy=True)



datagen.fit(train_X_Aug)



# Concatenar valores originales con augmented

train_X_Final = np.concatenate((train_X, train_X_Aug), axis=0)

train_Y_Final = np.concatenate((train_Y, train_Y_Aug), axis=0)
train_Y_Final = to_categorical(train_Y_Final, NUM_CLASSES).astype('int32')

val_Y = to_categorical(val_Y, NUM_CLASSES).astype('int32')
EPOCHS = 40

def graf_model(train_history):

    f = plt.figure(figsize=(EPOCHS,10))

    ax = f.add_subplot(121)

    ax2 = f.add_subplot(122)

    # summarize history for accuracy

    ax.plot(train_history.history['acc'])

    ax.plot(train_history.history['val_acc'])

    ax.set_title('model accuracy')

    ax.set_ylabel('accuracy')

    ax.set_xlabel('epoch')

    ax.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    ax2.plot(train_history.history['loss'])

    ax2.plot(train_history.history['val_loss'])

    ax2.set_title('model loss')

    ax2.set_ylabel('loss')

    ax2.set_xlabel('epoch')

    ax2.legend(['train', 'test'], loc='upper left')

    plt.show()
LEARNING_RATE = 0.00025

#LR_DECAY = LEARNING_RATE/EPOCHS

BATCH_SIZE = 32

L2_REG = 0.0030

LR_REDUC = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1,factor=0.5, min_lr=0.00001)
model = Sequential()



model.add(Conv2D(filters=32, kernel_size=7, padding="same", input_shape=(28, 28, 1), kernel_regularizer=l2(L2_REG)))

model.add(LeakyReLU(0.1))

model.add(Conv2D(filters=64, kernel_size=5, padding="same", kernel_regularizer=l2(L2_REG)))

model.add(LeakyReLU(0.1))

    

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.1))



model.add(Conv2D(filters=64, kernel_size=5, padding="valid", kernel_regularizer=l2(L2_REG)))

model.add(LeakyReLU(0.1))

model.add(Conv2D(filters=128, kernel_size=3, padding="valid", kernel_regularizer=l2(L2_REG)))

model.add(LeakyReLU(0.1))

    

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))

    

model.add(Flatten())

model.add(Dense(256, kernel_regularizer=l2(L2_REG)))

model.add(Dropout(0.30))

model.add(LeakyReLU(0.1))



model.add(Dense(128, kernel_regularizer=l2(L2_REG)))

model.add(Dropout(0.75))

model.add(LeakyReLU(0.1))

    

model.add(Dense(10))

model.add(Activation("softmax"))



#model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=LEARNING_RATE, decay=LR_DECAY), metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=LEARNING_RATE), metrics=['accuracy'])

model.summary()
# Data Augmentation ocurre en paralelo al entrenamiento

hist = model.fit_generator(datagen.flow(train_X_Final, train_Y_Final, batch_size=BATCH_SIZE),

                    steps_per_epoch=math.ceil(len(train_X) / BATCH_SIZE),

                    epochs = EPOCHS, 

                    validation_data = (val_X, val_Y),

                    callbacks=[LR_REDUC])

train_acc = hist.history['acc'][-1]

val_acc = hist.history['val_acc'][-1]

print('Train Acc: %.4f, Val Acc: %.4f' % (train_acc, val_acc))
graf_model(hist)
import sklearn.metrics as metrics



pred_Y = model.predict(val_X)  # shape=(n_samples, 12)

pred_Y = np.argmax(pred_Y, axis=1)  # only necessary if output has one-hot-encoding, shape=(n_samples)

true_Y = np.argmax(val_Y, axis=1)

confusion_matrix = metrics.confusion_matrix(y_true=true_Y, y_pred=pred_Y)  # shape=(12, 12)

print(confusion_matrix)
test = (test_df.values).reshape(-1, 28, 28 , 1)

pred = model.predict(test)

pred = np.argmax(pred,axis = 1)

pred = pd.Series(pred, name="Label")

submission = pd.concat([pd.Series(range(1 ,pred.shape[0]+1) ,name = "ImageId"), pred], axis = 1)

submission.to_csv("submission.csv",index=False)