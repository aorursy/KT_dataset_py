# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Get version python/keras/tensorflow/sklearn

from platform import python_version

import sklearn

import keras

import tensorflow as tf



# Folder manipulation

import os



# Garbage collector

import gc



# Linear algebra and data processing

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder



# Model evaluation

from sklearn.metrics import mean_absolute_error



# Visualisation of picture and graph

import matplotlib.pyplot as plt

import seaborn as sns



# Keras importation

from keras.applications import DenseNet201

from keras.models import Model, Sequential

from keras.optimizers import Adam

from keras_preprocessing.image import ImageDataGenerator

from keras.layers import BatchNormalization, Dense

from keras.regularizers import l1_l2

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.utils import to_categorical
print(os.listdir("../input"))

print("Keras version : " + keras.__version__)

print("Tensorflow version : " + tf.__version__)

print("Python version : " + python_version())

print("Sklearn version : " + sklearn.__version__)
MAIN_DIR = "../input/"

IMG_ROWS = 64

IMG_COLS = 64

CHANNELS = 3

IMG_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)

LABELS = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 

        'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 

        'Residential', 'River', 'SeaLake']



# Set graph font size

sns.set(font_scale=1.3)
def load_data():

    X_total_train = np.load(MAIN_DIR + "X_train.npy")

    X_train = X_total_train[:, :, :, :CHANNELS].copy()



    # Free memory

    del X_total_train

    gc.collect()

    

    y_total_train = np.load(MAIN_DIR + "y_train.npy")

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_total_train)



    # Free memory

    del y_total_train

    gc.collect()



    # One-hot encoding y_train

    y_train = to_categorical(y_train)



    # Free memory

    gc.collect()

    

    return X_train, y_train, encoder
X_train, y_train, encoder = load_data()

print(f"X shape : {X_train.shape}")

print(f"y shape : {y_train.shape}")
def plot_pictures(X, y, nb_rows=6, nb_cols=6, figsize=(14, 14)):

    # Set up the grid

    fig, ax = plt.subplots(nb_rows, nb_cols, figsize=figsize, gridspec_kw=None)

    fig.subplots_adjust(wspace=0.4, hspace=0.4)



    for i in range(0, nb_rows):

        for j in range(0, nb_cols):

            index = np.random.randint(0, X.shape[0])

    

            # Hide grid

            ax[i, j].grid(False)

            ax[i, j].axis('off')

            

            # Plot picture on grid

            ax[i, j].imshow(X[index].astype(np.int))

            ax[i, j].set_title(f"{LABELS[np.where(y[index] == 1)[0][0]]}")
plot_pictures(X_train, y_train)
def load_gen(X, y):

    gen = ImageDataGenerator(

            validation_split=0.1,

            horizontal_flip=True,

            vertical_flip=True,

            fill_mode='nearest')



    gen_train = gen.flow(X, y,

                          batch_size=32, 

                          shuffle=True,

                          subset='training')



    gen_val = gen.flow(X, y,

                         batch_size=32, 

                         shuffle=True, 

                         subset='validation')

    

    return gen_train, gen_val
def build_model():

    model = Sequential()

    

    model.add(DenseNet201(input_shape=IMG_SHAPE, include_top=False, pooling='max'))

    model.add(BatchNormalization())

    model.add(Dense(2048, activation='relu',  kernel_regularizer=l1_l2(0.01)))

    model.add(BatchNormalization())

    model.add(Dense(2048, activation='relu',  kernel_regularizer=l1_l2(0.01)))

    model.add(BatchNormalization())

    model.add(Dense(2048, activation='relu',  kernel_regularizer=l1_l2(0.01)))

    model.add(BatchNormalization())

    model.add(Dense(1024, activation='relu', kernel_regularizer=l1_l2(0.01)))

    model.add(BatchNormalization())

    model.add(Dense(10, activation='softmax'))

    

    for layer in model.layers:

        layer.trainable = True

    

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])

    

    model.summary()

    

    return model
def train_model(gen_train, gen_val):

    model = build_model()



    cbs = [ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, min_lr=1e-5, verbose=0),

           EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=10, verbose=1, mode='auto')]



    history = model.fit_generator(gen_train, 

                        steps_per_epoch=(gen_train.n//gen_train.batch_size), 

                        epochs=200, 

                        validation_data=gen_val, 

                        validation_steps=len(gen_val), 

                        shuffle=True, 

                        callbacks=cbs, 

                        verbose=1)

    return model, history
gc.collect()

gen_train, gen_val = load_gen(X_train, y_train)

model, history = train_model(gen_train, gen_val)
def plot_loss(history):

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    

    # Plot train/val accuracy

    ax[0].plot(history.history['acc'])

    ax[0].plot(history.history['val_acc'])

    ax[0].set_title('Model accuracy')

    ax[0].set_ylabel('Accuracy')

    ax[0].set_xlabel('Epochs')

    ax[0].legend(['Train', 'Test'], loc='lower right')

    ax[0].set_ylim(0, 1.05)

    

    # Plot train/val loss

    ax[1].plot(history.history['loss'])

    ax[1].plot(history.history['val_loss'])

    ax[1].set_title('Model Loss')

    ax[1].set_ylabel('Loss')

    ax[1].set_xlabel('Epochs')

    ax[1].legend(['Train', 'Test'], loc='upper right')
plot_loss(history)
def print_results(history):

    print("ACCURACY :")

    print(f"Training accuracy : {history.history['acc'][-1]}")

    print(f"Validation accuracy : {history.history['val_acc'][-1]}")

    

    print("\nLOSS :")

    print(f"Training categorical crossentropy loss : {history.history['loss'][-1]}")

    print(f"Validation categorical crossentropy loss : {history.history['val_loss'][-1]}")
print_results(history)
def plot_lr(history):

    fig, ax = plt.subplots(figsize=(7, 5))

    

    # Plot learning rate

    ax.plot(history.history['lr'])

    ax.set_title('Learning rate evolution')

    ax.set_ylabel('Learning rate value')

    ax.set_xlabel('Epochs')

    ax.legend(['Train'], loc='upper right')
plot_lr(history)
def load_data_test():

    X_total_test = np.load(MAIN_DIR + "X_test.npy")

    X_test = X_total_test[:, :, :, :CHANNELS].copy()



    # Free memory

    del X_total_test

    gc.collect()

    

    return X_test
X_test = load_data_test()



y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred,axis=1)
plot_pictures(X_test, to_categorical(y_pred))
def save_model(model):

    # Serialize model to JSON

    model_json = model.to_json()

    with open("model.json", "w") as json_file:

        json_file.write(model_json)

    

    # Serialize weights to HDF5

    model.save_weights("model.h5")

    print("Saved model to disk")
def save_pred(y_pred):

    y_pred = encoder.inverse_transform(y_pred)

    np.save("label_test.predict", y_pred)

    print("Save prediction")

    #!mv label_test.predict.npy label_test.predict

    #!zip -r submission.zip label_test.predict
save_pred(y_pred)

save_model(model)