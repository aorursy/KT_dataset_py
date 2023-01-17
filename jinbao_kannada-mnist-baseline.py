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
%%time

train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
train_data.groupby(by='label').size()
IMG_SIZE = 28
from keras.utils import to_categorical

img_train = train_data.drop(["label"], axis=1).values.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32')

img_label = to_categorical(train_data["label"])



img_test = test_data.drop(["id"], axis=1).values.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32')



print("img_train.shape = ", img_train.shape)

print("img_label.shape = ", img_label.shape)

print("img_test.shape = ", img_test.shape)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(img_train, img_label, test_size=0.15)

print("x_train.shape = ", x_train.shape)

print("y_train.shape = ", y_train.shape)

print("x_test.shape = ", x_test.shape)

print("y_test.shape = ", y_test.shape)
import keras

from keras.datasets import mnist

from keras.layers import Input, Dense, Dropout, Flatten, add

from keras.layers import Conv2D, Activation, MaxPooling2D, AveragePooling2D, BatchNormalization

from keras import backend as K

from keras.callbacks import ModelCheckpoint

import tensorflow as tf

from keras.models import Model

from keras.utils import plot_model

from keras.preprocessing. image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input, Dense, PReLU, Dropout

from keras.models import Model

from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

from keras.optimizers import SGD, Adam
# model from https://www.kaggle.com/anshumandec94/6-layer-conv-nn-using-adam

def build_model(input_shape=(28, 28, 1), classes = 10):

    input_layer = Input(shape=input_shape)

    x = Conv2D(16, (3,3), strides=1, padding="same", name="conv1")(input_layer)

    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform", name="batch1")(x)

    x = Activation('relu',name='relu1')(x)

    x = Dropout(0.1)(x)

    

    x = Conv2D(32, (3,3), strides=1, padding="same", name="conv2")(x)

    x = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform", name="batch2")(x)

    x = Activation('relu',name='relu2')(x)

    x = Dropout(0.15)(x)

    x = MaxPooling2D(pool_size=2, strides=2, padding="same", name="max2")(x)

    

    x = Conv2D(64, (5,5), strides=1, padding ="same", name="conv3")(x)

    x = BatchNormalization(momentum=0.17, epsilon=1e-5, gamma_initializer="uniform", name="batch3")(x)

    x = Activation('relu', name="relu3")(x)

    x = MaxPooling2D(pool_size=2, strides=2, padding="same", name="max3")(x)

    

    x = Conv2D(128, (5,5), strides=1, padding="same", name="conv4")(x)

    x = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform", name="batch4")(x)

    x = Activation('relu', name="relu4")(x)

    x = Dropout(0.17)(x)

    

    x = Conv2D(64, (3,3), strides=1, padding="same", name="conv5")(x)

    x = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform", name="batch5")(x)

    x = Activation('relu', name='relu5')(x)

    x = Dropout(0.2)(x)

    

    x = Conv2D(32, (3,3), strides=1, padding="same", name="conv6")(x)

    x = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform", name="batch6" )(x)

    

    x = Activation('relu', name="relu6")(x)

    x = Dropout(0.05)(x)

    

    x = Flatten()(x)

    x = Dense(50, name="Dense1")(x)

    x = Activation('relu', name='relu7')(x)

    x = Dropout(0.05)(x)

    x = Dense(25, name="Dense2")(x)

    x = Activation('relu', name='relu8')(x)

    x = Dropout(0.03)(x)

    x = Dense(classes, name="Dense3")(x)

    x = Activation('softmax')(x)



    model = Model(inputs=input_layer, outputs=x)

    return model
model = build_model(input_shape=(28, 28, 1), classes = 10)
train_datagen = ImageDataGenerator(

    rotation_range=9, 

    zoom_range=0.25, 

    width_shift_range=0.25, 

    height_shift_range=0.25,

    rescale=1./255

)

train_datagen.fit(x_train)

test_datagen = ImageDataGenerator(rescale=1./255)

adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.5, min_lr=0.00001)

checkpoint = ModelCheckpoint("bestmodel.model", monitor='val_acc', verbose=1, save_best_only=True)

earlyStopping = EarlyStopping(monitor='val_acc', patience=15, verbose=1, mode='min')
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
epochs = 80

batch_size = 128
history = model.fit_generator(

    train_datagen.flow(x_train, y_train, batch_size=batch_size),

    steps_per_epoch=x_train.shape[0] // batch_size,

    epochs=epochs,

    validation_data=test_datagen.flow(x_test, y_test),

    validation_steps=x_test.shape[0] // batch_size,

    callbacks=[checkpoint, learning_rate_reduction])
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

def PlotLoss(his, epoch):

    plt.style.use("ggplot")

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history["loss"], label="train_loss")

    plt.plot(np.arange(0, epoch), his.history["val_loss"], label="val_loss")

    plt.title("Training Loss")

    plt.xlabel("Epoch #")

    plt.ylabel("Loss")

    plt.legend(loc="upper right")

    plt.show()



def PlotAcc(his, epoch):

    plt.style.use("ggplot")

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history["acc"], label="train_acc")

    plt.plot(np.arange(0, epoch), his.history["val_acc"], label="val_acc")

    plt.title("Training Accuracy")

    plt.xlabel("Epoch #")

    plt.ylabel("Accuracy")

    plt.legend(loc="upper right")

    plt.show()
PlotAcc(history, epochs)

PlotLoss(history, epochs)
model.load_weights('bestmodel.model')
results=model.predict(img_test/255.0)
x2_train, x2_test, y2_train, y2_test = train_test_split(img_test, results, test_size=0.15)
x_train_final = np.concatenate((img_train,x2_train), axis=0)

y_train_final = np.concatenate((img_label,y2_train), axis=0)
x_train, x_test, y_train, y_test = train_test_split(x_train_final, y_train_final, test_size=0.15)

print("x_train.shape = ", x_train.shape)

print("y_train.shape = ", y_train.shape)

print("x_test.shape = ", x_test.shape)

print("y_test.shape = ", y_test.shape)
keras.backend.clear_session()
model2 = build_model(input_shape=(28, 28, 1), classes = 10)
sgd2 = SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)

learning_rate_reduction2 = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.5, min_lr=0.00001)

checkpoint2 = ModelCheckpoint("bestmodel2.model", monitor='val_acc', verbose=1, save_best_only=True)
model2.compile(loss='categorical_crossentropy', optimizer=sgd2, metrics=['acc'])
history2 = model2.fit_generator(

    train_datagen.flow(x_train, y_train, batch_size=batch_size),

    steps_per_epoch=x_train.shape[0] // batch_size,

    epochs=epochs,

    validation_data=test_datagen.flow(x_test, y_test),

    validation_steps=x_test.shape[0] // batch_size,

    callbacks=[checkpoint2, learning_rate_reduction2])
PlotAcc(history2, epochs)

PlotLoss(history2, epochs)
#model2.load_weights('bestmodel2.model')
results=model2.predict(img_test/255.0)

results=np.argmax(results, axis=1)

sub=pd.DataFrame()

sub['id']=list(test_data.values[0:,0])

sub['label']=results
sub.head()
sub.to_csv("submission.csv", index=False)