import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import warnings, os



from tqdm import tqdm



from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow.keras import Model

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalMaxPooling2D, BatchNormalization,Input

from tensorflow.keras.optimizers import RMSprop, Adam

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.vgg16 import VGG16

warnings.filterwarnings('ignore')
def plot_model_result(history):

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title("Model Loss")

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend(['Train', 'Test'])

    plt.show()
BATCH_SIZE = 256

EPOCHS = 20

IMAGE_SIZE = 28
dftrain = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

dftest = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

dftrain.shape, dftest.shape, dftrain.columns
fig,ax = plt.subplots(1,5,figsize=(20,5))

for i in range(5):

    ax[i].imshow(dftrain.iloc[i,1:].values.reshape(IMAGE_SIZE,IMAGE_SIZE))

plt.show()
# w * x = b

x, y = tf.constant(3.0), tf.constant(6.0)

w = tf.Variable(10.0)

loss = tf.math.abs(w * x - y)

w,x,y,loss
def train_step():

    with tf.GradientTape() as tape:

        loss = tf.math.abs(w * x - y)

        dw = tape.gradient(loss, w)

        print('w = {:.2f}, dw = {:2f}'.format(w.numpy(), dw))

        w.assign(w - dw)



for i in range(8):

    train_step()

NUM_CLASS = dftrain['label'].nunique()

X = dftrain.iloc[:,1:].values.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1).astype(float)

y = to_categorical(dftrain['label'].values, num_classes=NUM_CLASS)

X.shape, y.shape
model = Sequential()

model.add(Input(shape=(IMAGE_SIZE,IMAGE_SIZE,1)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(NUM_CLASS, activation='softmax'))

model.compile(loss=categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

model.summary()
history = model.fit(X, y,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_split = 0.2)
plot_model_result(history)
model = Sequential()

model.add(Conv2D(20, kernel_size=3,activation='relu',input_shape=(IMAGE_SIZE,IMAGE_SIZE,1)))

model.add(BatchNormalization())

model.add(Conv2D(20, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(NUM_CLASS, activation='softmax'))

model.compile(loss=categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

model.summary()
history = model.fit(X, y,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_split = 0.2)
plot_model_result(history)
model = Sequential()

model.add(Conv2D(32, kernel_size=3,activation='relu',padding='same', input_shape=(IMAGE_SIZE,IMAGE_SIZE,1)))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, padding='same'))

model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, padding='same'))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))

model.add(Dense(NUM_CLASS, activation='softmax'))

model.compile(loss=categorical_crossentropy,optimizer=Adam(lr=1e-3),metrics=['accuracy'])

model.summary()
history = model.fit(X, y,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_split = 0.2)
plot_model_result(history)
X_train,X_val,y_train,y_val = train_test_split(X,y, stratify=y)

X_train.shape,y_train.shape,X_val.shape,y_val.shape
datagen = ImageDataGenerator(

        rotation_range = 10,

        horizontal_flip=True)





datagen.fit(X_train)
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=(IMAGE_SIZE,IMAGE_SIZE,1)))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))

model.add(MaxPool2D(pool_size=2))

model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same',))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(loss=categorical_crossentropy,optimizer=Adam(lr=1e-3),metrics=['accuracy'])

model.summary()
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, min_lr=1e-8, factor=0.1),

             EarlyStopping(monitor='val_loss', patience=10, verbose=1, min_delta=0.001),

             ModelCheckpoint('model.h5', save_best_only=True, save_weights_only=True)]
history = model.fit_generator(generator=datagen.flow(X_train,y_train,batch_size=BATCH_SIZE),

                                epochs=EPOCHS,

                                steps_per_epoch=X_train.shape[0] // BATCH_SIZE,

                                callbacks=callbacks,

                                verbose=1,

                                validation_data=(X_val,y_val),

                                validation_steps =X_val.shape[0] // BATCH_SIZE)
plot_model_result(history)
base_model=VGG16(include_top=True, weights=None,input_shape=(32,32,1), pooling='avg')

base_model.summary()
base_model = VGG16(include_top=False, weights=None,input_shape=(32,32,1), pooling='avg')

base_model.summary()
model=Sequential()

model.add(base_model)

model.layers[0].trainable = False

model.add(Dense(256,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.summary()