# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Get version python/keras/tensorflow

from platform import python_version

import keras

import tensorflow as tf



# Folder manipulation

from pathlib import Path

import os



# Linear algebra and data processing

import numpy as np

import pandas as pd



# Model evaluation

from sklearn.metrics import confusion_matrix



# Visualisation of picture and graph

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

from PIL import Image



# For loading bar

import tqdm



# Keras importation

from tqdm import tqdm, tqdm_notebook

from keras.models import Sequential

from keras.optimizers import Adam

from keras_preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dense, Dropout, Flatten

from keras.regularizers import l2

from keras.callbacks import ReduceLROnPlateau, EarlyStopping
print(os.listdir("../input"))

print("Keras version : " + keras.__version__)

print("Tensorflow version : " + tf.__version__)

print("Python version : " + python_version())
MAIN_DIR = '../input/'

TRAIN_DIR = MAIN_DIR + "train/train/"

TEST_DIR = MAIN_DIR + "test/test/"

IMG_ROWS = 32

IMG_COLS = 32

CHANNELS = 3

IMG_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)



# Set graph font size

sns.set(font_scale=1.3)
def plot_pictures(nb_rows=6, nb_cols=6, figsize=(14, 14)):

    # Set up the grid

    fig, ax = plt.subplots(nb_rows, nb_cols, figsize=figsize, gridspec_kw=None)

    fig.subplots_adjust(wspace=0.4, hspace=0.4)



    for i in range(0, nb_rows):

        for j in range(0, nb_cols):

            data = pd.read_csv(MAIN_DIR + "train.csv")

            index = np.random.randint(0, data.shape[0])

            file = data.loc[index]['id']



            # Load picture

            img_ = cv2.imread(TRAIN_DIR + file)

    

            # Hide grid

            ax[i, j].grid(False)

            ax[i, j].axis('off')

            

            # Plot picture on grid

            ax[i, j].imshow(img_)

            ax[i, j].set_title("Label : " + str(data.loc[index]['has_cactus']))
plot_pictures(6, 6)
df_train = pd.read_csv(MAIN_DIR + "train.csv")
fig, ax = plt.subplots(figsize=(5, 5))

sns.countplot(df_train['has_cactus'])
df_train['has_cactus'].value_counts()
def load_data(dataframe=None, batch_size=16, mode='categorical'):

    gen = ImageDataGenerator(

        zoom_range=0.1,

        validation_split=0.1,

        horizontal_flip=True,

        vertical_flip=True,

        fill_mode='nearest')

    

    df = pd.read_csv(MAIN_DIR + 'train.csv')

    df['has_cactus'] = df['has_cactus'].apply(str)

    

    data_train = gen.flow_from_dataframe(df, 

                                          directory=TRAIN_DIR, 

                                          x_col='id', 

                                          y_col='has_cactus', 

                                          has_ext=True, 

                                          target_size=(32, 32),

                                          class_mode=mode, 

                                          batch_size=batch_size, 

                                          shuffle=True,

                                          subset='training')

        

    data_test = gen.flow_from_dataframe(df, 

                                      directory=TRAIN_DIR, # Changer les chemins

                                      x_col='id', 

                                      y_col='has_cactus', 

                                      has_ext=True, 

                                      target_size=(32, 32),

                                      class_mode=mode, 

                                      batch_size=batch_size, 

                                      shuffle=True, 

                                      subset='validation')

    

    return data_train, data_test
def create_model():

    model = Sequential()

    

    model.add(Conv2D(16, (3, 3), input_shape=(32, 32, 3), padding='same', use_bias=False, name="first_conv"))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False, name="second_conv"))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.25))



    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.25))



    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    

    model.add(Conv2D(256, (3, 3), padding='same', use_bias=False))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), padding='same', use_bias=False))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    

    model.add(Conv2D(512, (3, 3), padding='same', use_bias=False))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', use_bias=False))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Dense(2, activation='softmax'))

    

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])

    

    return model
def train(train_generator, val_generator):

    model = create_model()



    cbs = [ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, min_lr=1e-7, verbose=0),

           EarlyStopping(monitor='val_loss', min_delta=0.0001, 

                         patience=10, verbose=1, mode='min', restore_best_weights=True)]

    

    history = model.fit_generator(train_generator, 

                        steps_per_epoch=(train_generator.n//train_generator.batch_size), 

                        epochs=100, 

                        validation_data=val_generator, 

                        validation_steps=len(val_generator), 

                        shuffle=True, 

                        callbacks=cbs, 

                        verbose=1)

    return model, history
train_generator, val_generator = load_data(batch_size=32)

model, history = train(train_generator, val_generator)
model.summary()
def plot_loss(history):

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    

    # Plot train/val accuracy

    ax[0].plot(history.history['acc'])

    ax[0].plot(history.history['val_acc'])

    ax[0].set_title('Model accuracy')

    ax[0].set_ylabel('Accuracy')

    ax[0].set_xlabel('Epochs')

    ax[0].legend(['Train', 'Test'], loc='upper left')

    

    # Plot train/val loss

    ax[1].plot(history.history['loss'])

    ax[1].plot(history.history['val_loss'])

    ax[1].set_title('Model Loss')

    ax[1].set_ylabel('Loss')

    ax[1].set_xlabel('Epochs')

    ax[1].legend(['Train', 'Test'], loc='upper left')
plot_loss(history)
def plot_lr(history):

    fig, ax = plt.subplots(figsize=(7, 5))

    

    # Plot learning rate

    ax.plot(history.history['lr'])

    ax.set_title('Learning rate evolution')

    ax.set_ylabel('Learning rate')

    ax.set_xlabel('Epochs')

    ax.legend(['Train', 'Test'], loc='upper left')
plot_lr(history)
# Code inspire from : https://gist.github.com/oeway/f0ed87d3df671b351b533108bf4d9d5d

def plot_conv_weights(model, layer):

    W = model.get_layer(name=layer).get_weights()[0]

    if len(W.shape) == 4:

        W = np.squeeze(W)

        W = W.reshape((W.shape[0], W.shape[1], W.shape[2]*W.shape[3])) 

        fig, axs = plt.subplots(5,5, figsize=(9,8))

        fig.subplots_adjust(hspace = .5, wspace=.001)

        axs = axs.ravel()

        for i in range(25):

            im = axs[i].imshow(W[:,:,i], cmap=plt.cm.get_cmap('Blues', 6))

            axs[i].set_title(str(i))

            

            # Hide grid

            axs[i].grid(False)

            axs[i].axis('off')

            

    fig.subplots_adjust(right=0.8)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])

    fig.colorbar(im, cax=cbar_ax)
plot_conv_weights(model, "first_conv")
plot_conv_weights(model, "second_conv")
def predict(model, sample_submission):

    pred = np.empty((sample_submission.shape[0],))

    for n in tqdm(range(sample_submission.shape[0])):

        data = np.array(Image.open(TEST_DIR + sample_submission.id[n]))

        pred[n] = model.predict(data.reshape((1, 32, 32, 3)))[0][1]

    

    sample_submission['has_cactus'] = pred

    return sample_submission
sample_submission = pd.read_csv(MAIN_DIR + 'sample_submission.csv')

df_prediction = predict(model, sample_submission)



df_prediction.to_csv('submission.csv', index=False)
df_prediction.head()