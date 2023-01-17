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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy

from IPython.display import SVG
import IPython.display as display

import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import plot_model
from keras.utils import model_to_dot
dfTrain = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
dfTest  = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

X     = dfTrain[dfTrain.columns[1:]].to_numpy().reshape(42000,28*28)
y     = to_categorical(dfTrain[dfTrain.columns[0]].to_numpy())
#y     = dfTrain[dfTrain.columns[0]].to_numpy()

train_X, valid_X, train_y, valid_y = train_test_split(X,y,shuffle=False, random_state=1,train_size=0.95)
test_X = dfTest.to_numpy().reshape(28000,28*28)

print(X.shape,y.shape)
print(train_X.shape,valid_X.shape,train_y.shape,valid_y.shape)
print(test_X.shape)
xPoints = np.linspace(-5,5,1000)
yPoints = 1/(1+np.exp(-xPoints))
fig, ax = plt.subplots(1,2,figsize=(15,5))
ax[0].plot(xPoints,yPoints,label='sigmoid')
yPoints = np.max([np.zeros(len(xPoints)),xPoints],axis=0)
ax[0].plot(xPoints,yPoints,label='relu')
ax[0].set_ylim(-0.5,2)
ax[0].grid(True)
ax[0].legend()

labels  = np.linspace(0,9,10)
zPoints = np.random.random(10)
zPoints[5]=3
ax[1].bar(labels-0.1,zPoints,width=0.2,label='original numbers')
ax[1].bar(labels+0.1,scipy.special.softmax(zPoints),width=0.2,label='softmax numbers')
ax[1].legend()
modelA = tf.keras.Sequential(
  [
    tf.keras.layers.Input(shape=(28*28,)),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

modelB = tf.keras.Sequential(
  [
    tf.keras.layers.Input(shape=(28*28,)),
    tf.keras.layers.Dense(200, activation='sigmoid'),
    tf.keras.layers.Dense(100, activation='sigmoid'),
    tf.keras.layers.Dense(60, activation='sigmoid'),
    tf.keras.layers.Dense(30, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

modelC = tf.keras.Sequential(
  [
    tf.keras.layers.Input(shape=(28*28,)),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

modelD = tf.keras.Sequential(
  [
    tf.keras.layers.Input(shape=(28*28,)),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

modelA.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelB.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelC.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelD.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
BATCH_SIZE = 100
EPOCHS = 30

steps_per_epoch = 42000//BATCH_SIZE  # 60,000 items in this dataset
print("Steps per epoch: ", steps_per_epoch)
historyA = modelA.fit(train_X, train_y, 
                    steps_per_epoch=steps_per_epoch, 
                    epochs=EPOCHS,
                    validation_data=(valid_X, valid_y), 
                    validation_steps=1)

historyB = modelB.fit(train_X, train_y, 
                    steps_per_epoch=steps_per_epoch, 
                    epochs=EPOCHS,
                    validation_data=(valid_X, valid_y), 
                    validation_steps=1)

historyC = modelC.fit(train_X, train_y, 
                    steps_per_epoch=steps_per_epoch, 
                    epochs=EPOCHS,
                    validation_data=(valid_X, valid_y), 
                    validation_steps=1)

historyD = modelD.fit(train_X, train_y, 
                    steps_per_epoch=steps_per_epoch, 
                    epochs=EPOCHS,
                    validation_data=(valid_X, valid_y), 
                    validation_steps=1)
fig, ax = plt.subplots(1,2,figsize=(15,5))
ax[0].plot(historyA.history['accuracy'],    color='red', linestyle='-',label='1 layer (10) softmax')
ax[0].plot(historyA.history['val_accuracy'],color='blue',linestyle='-')
ax[0].plot(historyB.history['accuracy'],    color='red', linestyle='-.',label='5 layers (200/100/60/30/10) sigmoid/softmax')
ax[0].plot(historyB.history['val_accuracy'],color='blue',linestyle='-.')
ax[0].plot(historyC.history['accuracy'],    color='red', linestyle='--',label='5 layers (200/100/60/30/10) relu/softmax')
ax[0].plot(historyC.history['val_accuracy'],color='blue',linestyle='--')
ax[0].plot(historyD.history['accuracy'],    color='red', linestyle=':',label='5 layers (200/100/60/30/10) relu/softmax + dropout')
ax[0].plot(historyD.history['val_accuracy'],color='blue',linestyle=':')
ax[0].set_title('Model accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylim(0.75,1.0)
ax[0].legend()
ax[0].grid(True)

# Plot training & validation loss values
ax[1].semilogy(historyA.history['loss'],    color='red', linestyle='-',label='1 layer (10) softmax')
ax[1].semilogy(historyA.history['val_loss'],color='blue',linestyle='-')
ax[1].semilogy(historyB.history['loss'],    color='red', linestyle='-.',label='5 layers (200/100/60/30/10) sigmoid/softmax')
ax[1].semilogy(historyB.history['val_loss'],color='blue',linestyle='-.')
ax[1].semilogy(historyC.history['loss'],    color='red', linestyle='--',label='5 layers (200/100/60/30/10) relu/softmax')
ax[1].semilogy(historyC.history['val_loss'],color='blue',linestyle='--')
ax[1].semilogy(historyD.history['loss'],    color='red', linestyle=':',label='5 layers (200/100/60/30/10) relu/softmax + dropout')
ax[1].semilogy(historyD.history['val_loss'],color='blue',linestyle=':')
ax[1].set_title('Model loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend()
ax[1].grid(True)
BATCH_SIZE = 100
EPOCHS = 30

steps_per_epoch = 42000//BATCH_SIZE

modelE = tf.keras.Sequential(
  [
    tf.keras.layers.Reshape(input_shape=(28*28,), target_shape=(28,28,1)),
    tf.keras.layers.Conv2D(kernel_size=6, filters=6,  strides=1, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(kernel_size=5, filters=12, strides=2, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(kernel_size=4, filters=24, strides=2, padding='same', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

modelF = tf.keras.Sequential(
  [
    tf.keras.layers.Reshape(input_shape=(28*28,), target_shape=(28,28,1)),
    tf.keras.layers.Conv2D(kernel_size=5, filters=24, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(kernel_size=5, filters=48, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(kernel_size=5, filters=64, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

modelE.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelF.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
historyE = modelE.fit(train_X, train_y, 
                    steps_per_epoch=steps_per_epoch, 
                    epochs=EPOCHS,
                    validation_data=(valid_X, valid_y), 
                    validation_steps=1)

historyF = modelF.fit(train_X, train_y, 
                    steps_per_epoch=steps_per_epoch, 
                    epochs=EPOCHS,
                    validation_data=(valid_X, valid_y), 
                    validation_steps=1)
fig, ax = plt.subplots(1,2,figsize=(15,5))
ax[0].plot(historyE.history['accuracy'],    color='red', linestyle='-')
ax[0].plot(historyE.history['val_accuracy'],color='blue',linestyle='-')
ax[0].plot(historyF.history['accuracy'],    color='red', linestyle='-.')
ax[0].plot(historyF.history['val_accuracy'],color='blue',linestyle='-.')
ax[0].set_title('Model accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylim(0.90,1.0)
ax[0].legend()
ax[0].grid(True)

# Plot training & validation loss values
ax[1].semilogy(historyE.history['loss'],    color='red', linestyle='-')
ax[1].semilogy(historyE.history['val_loss'],color='blue',linestyle='-')
ax[1].semilogy(historyF.history['loss'],    color='red', linestyle='-.')
ax[1].semilogy(historyF.history['val_loss'],color='blue',linestyle='-.')
ax[1].set_title('Model loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend()
ax[1].grid(True)
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation
from keras.layers import Add, Flatten, AveragePooling2D, Dense, Dropout
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import time


def residual_block(inputs, filters, strides=1):
    """Residual block
    
    Shortcut after Conv2D -> ReLU -> BatchNorm -> Conv2D
    
    Arguments:
        inputs (tensor): input
        filters (int): Conv2D number of filterns
        strides (int): Conv2D square stride dimensions

    Returns:
        x (tensor): input Tensor for the next layer
    """
    y = inputs # Shortcut path
    
    # Main path
    x = Conv2D(kernel_size=3,filters=filters,strides=strides,padding='same',)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(kernel_size=3,filters=filters,strides=1,padding='same',)(x)
    x = BatchNormalization()(x)
    
    # Fit shortcut path dimenstions
    if strides > 1:
        y = Conv2D(kernel_size=3,filters=filters,strides=strides,padding='same',)(y)
        y = BatchNormalization()(y)
    
    # Concatenate paths
    x = Add()([x, y])
    x = Activation('relu')(x)
    
    return x
    
    
def resnet(input_shape, num_classes, filters, stages):
    """ResNet 
    
    At the beginning of each stage downsample feature map size 
    by a convolutional layer with strides=2, and double the number of filters.
    The kernel size is the same for each residual block.
    
    Arguments:
        input_shape (3D tuple): shape of input Tensor
        filters (int): Conv2D number of filterns
        stages (1D list): list of number of resiual block in each stage eg. [2, 5, 5, 2]
    
    Returns:
        model (Model): Keras model
    """
    # Start model definition
    inputs = Input(shape=input_shape)
    x = Conv2D(kernel_size=7,filters=filters,strides=1,padding='same',)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Stack residual blocks
    for stage in stages:
        x = residual_block(x, filters, strides=2)
        for i in range(stage-1):
            x = residual_block(x, filters)
        filters *= 2
        
    # Pool -> Flatten -> Classify
    x = AveragePooling2D(4)(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(int(filters/4), activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Instantiate model
    model = Model(inputs=inputs, outputs=outputs)
    return model 


# Reshape and normalize
X = dfTrain.drop(columns=['label']).values.reshape(-1, 28, 28, 1) / 255
y = dfTrain['label'].values

# Get training and testing datasets
X_train, X_valid, y_train, y_valid = train_test_split(X, y,shuffle=False, random_state=1,train_size=0.95)
X_test = dfTest.values.reshape(-1, 28, 28, 1) / 255

print(X.shape,y.shape)
print(X_train.shape,X_valid.shape,y_train.shape,y_valid.shape)
print(test_X.shape)

epochs=10
filters=64
stages=[3, 3, 3]
batch_size=128

modelG = resnet(input_shape=X[0].shape,num_classes=np.unique(y).shape[-1],filters=filters,stages=stages)
modelH = resnet(input_shape=X[0].shape,num_classes=np.unique(y).shape[-1],filters=filters,stages=stages)
modelG.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
modelH.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# Define callbacks
checkpoint = ModelCheckpoint(
    filepath=f'resnet-{int(time.time())}.dhf5',
    monitor='loss',
    save_best_only=True
)

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8**x)
callbacks = [checkpoint, annealer]

# Define data generator
datagen = ImageDataGenerator(  
    rotation_range=10,  
    zoom_range=0.1, 
    width_shift_range=0.1, 
    height_shift_range=0.1
)
datagen.fit(X)
historyG = modelG.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=epochs, 
    verbose=1,
    callbacks=callbacks)
historyH = modelH.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_valid, y_valid),
    epochs=epochs, 
    verbose=1,
    callbacks=callbacks)
fig, ax = plt.subplots(1,2,figsize=(15,5))
ax[0].plot(historyG.history['accuracy'],    color='red', linestyle='-',label='resNet')
ax[0].plot(historyG.history['val_accuracy'],color='blue',linestyle='-')
ax[0].plot(historyH.history['accuracy'],    color='red', linestyle='-.',label='resNet + enhanced data')
ax[0].plot(historyH.history['val_accuracy'],color='blue',linestyle='-.')
ax[0].set_title('Model accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylim(0.90,1.0)
ax[0].legend()
ax[0].grid(True)

# Plot training & validation loss values
ax[1].semilogy(historyG.history['loss'],    color='red', linestyle='-',label='resNet')
ax[1].semilogy(historyG.history['val_loss'],color='blue',linestyle='-')
ax[1].semilogy(historyH.history['loss'],    color='red', linestyle='-.',label='resNet + enhanced data')
ax[1].semilogy(historyH.history['val_loss'],color='blue',linestyle='-.')
ax[1].set_title('Model loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend()
ax[1].grid(True)
def writeSubmission(modelDict, test):
    for k in modelDict.keys():
        model = modelDict[k]
        output     = model.predict(test)
        prediction = np.argmax(output,axis=1)

        dfOut = pd.DataFrame(data={'ImageId':np.arange(1,28001,1),'Label':prediction})
        dfOut.to_csv('submission'+k+'.csv', index=False)
        if k == 'H':
            dfOut.to_csv('submission.csv', index=False)
        
modelDict1 = {'A':modelA, 'B':modelB, 'C':modelC, 'D':modelD, 'E':modelE, 'F':modelF}
modelDict2 = {'G':modelG, 'H':modelH}

writeSubmission(modelDict1, test_X)
writeSubmission(modelDict2, X_test)
