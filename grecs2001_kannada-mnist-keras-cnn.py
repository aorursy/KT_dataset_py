# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Linear algebra and data processing

import numpy as np

import pandas as pd

import math

import random



# Get version python/keras/tensorflow/sklearn

from platform import python_version

import sklearn

import keras

import tensorflow as tf



# Folder manipulation

import os



# Spliting data

from sklearn.model_selection import train_test_split



# Keras importation

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.optimizers import Adam

from keras.models import Model, Sequential

from keras.layers import Input, Conv2D, Flatten, MaxPooling2D

from keras.layers.core import Dense, Dropout, Activation

from keras.layers.normalization import BatchNormalization



# For images augmentations

import albumentations as albu



# Visualizaton

import matplotlib.pyplot as plt

import seaborn as sns
print(os.listdir("../input"))

print("Keras version : " + keras.__version__)

print("Tensorflow version : " + tf.__version__)

print("Python version : " + python_version())

print("Sklearn version : " + sklearn.__version__)
MAIN_DIR = "../input/Kannada-MNIST/"



DATA_TRAIN = MAIN_DIR + "train.csv"

DATA_TEST = MAIN_DIR + "test.csv"



IMG_HEIGHT = 28

IMG_WIDTH = 28

CHANNELS = 1

IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, CHANNELS)



# Set graph font size

sns.set(font_scale=1.3)
def load_data():

    df_train =  pd.read_csv(DATA_TRAIN)

    df_test =  pd.read_csv(DATA_TEST)

    return df_train, df_test
data_train, data_test = load_data()
print(f"Training data has shape : {data_train.shape}")

print(f"Test data has shape : {data_test.shape}")
data_train.head()
data_test.head()
print(f"Data train has {data_train.isna().sum().sum(axis=0)} Nan values")

print(f"Data test has {data_test.isna().sum().sum(axis=0)} Nan values")
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

sns.countplot(x='label', data=data_train, ax=ax)
def plot_images(data, nb_rows=6, nb_cols=6, figsize=(14, 14)):

    # Get data

    df = data.copy()

    X_raw = df.drop(['label'], axis=1).values

    X = np.reshape(X_raw, (X_raw.shape[0], IMG_WIDTH, IMG_HEIGHT))

    y = df['label'].values

        

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

            ax[i, j].imshow(X[index].astype(np.int), cmap='gray')

            ax[i, j].set_title(f"Label : {y[index]}")
plot_images(data_train)
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self,

                 data,

                 list_IDs, 

                 labels=None,

                 batch_size=32,

                 dim=IMG_SHAPE,

                 n_channels=1,

                 augment=False,

                 n_classes=10,

                 mode='fit',

                 shuffle=True,

                 random_state=42):

        

        'Initialization'

        self.data = data

        self.dim = dim

        self.batch_size = batch_size

        self.labels = labels

        self.list_IDs = list_IDs

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.augment = augment

        self.mode = mode

        self.random_state = random_state

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        if(self.mode == 'fit'):

            return int(np.floor(len(self.list_IDs) / self.batch_size))

        else:

            return int(np.ceil(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Augment 1 batch over 2

        augment_batch = random.choice([True, False])

        

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        list_IDs_batch = [self.list_IDs[k] for k in indexes]



        # Generate input

        X = self.__generate_X(list_IDs_batch)

        

        # Generate target or predict        

        if(self.mode == 'fit'):

            y = self.__generate_y(list_IDs_batch)

            

            if(self.augment and augment_batch):

                X = self.__augment_batch(X)

            

            return X, y

        

        elif self.mode == 'predict':

            return X

        else:

            raise AttributeError('The mode parameter should be set to "fit" or "predict".')



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.seed(self.random_state)

            np.random.shuffle(self.indexes)



    def __generate_X(self, list_IDs_batch):

        'Generates data containing batch_size samples'

        # Initialization

        X = np.empty((len(list_IDs_batch), *self.dim))

        

        # Generate data

        for i, ID in enumerate(list_IDs_batch):

            # Store sample

            X[i,] = np.reshape(self.data[ID], self.dim)



        return X

    

    def __generate_y(self, list_IDs_batch):

        y = np.empty((self.batch_size), dtype=int)

        

        for i, ID in enumerate(list_IDs_batch):

            # Generate data

            y[i, ] = self.labels[ID]



        return keras.utils.to_categorical(y, num_classes=self.n_classes)

    

    def __random_transform(self, img):

        composition = albu.Compose([

            albu.ShiftScaleRotate(rotate_limit=10, shift_limit=0.15, scale_limit=0.1)

        ])

        

        composed = composition(image=img)        

        aug_img = composed['image']

        

        return aug_img

    

    def __augment_batch(self, img_batch):

        for i in range(img_batch.shape[0]):

            img_batch[i, ] = self.__random_transform(img_batch[i, ])

        

        return img_batch
def build_model():

    input_layer = Input(shape=IMG_SHAPE)

    

    x = Conv2D(32, (3,3), strides=1, padding="same", name="conv1")(input_layer)

    x = BatchNormalization(name="batch1")(x)

    x = Activation('relu', name='relu1')(x)

    x = MaxPooling2D(pool_size=2, strides=2, padding="valid", name="max2")(x)

    x = Conv2D(32, (3, 3), padding='same', name="conv2")(x)

    x = BatchNormalization(name="batch2")(x)

    x = Activation('relu', name='relu2')(x)

    x = Dropout(0.4, name='dropout1')(x)

    

    x = Conv2D(64, (3,3), strides=1, padding="same", name="conv3")(x)

    x = BatchNormalization(name="batch3")(x)

    x = Activation('relu', name='relu3')(x)

    x = MaxPooling2D(pool_size=2, strides=2, padding="valid", name="max3")(x)

    x = Conv2D(64, (3, 3), padding='same', name="conv4")(x)

    x = BatchNormalization(name="batch4")(x)

    x = Activation('relu', name='relu4')(x)

    x = Dropout(0.4, name='dropout2')(x)

    

    x = Conv2D(128, (3,3), strides=1, padding="same", name="conv5")(x)

    x = BatchNormalization(name="batch5")(x)

    x = Activation('relu', name='relu5')(x)

    x = MaxPooling2D(pool_size=2, strides=2, padding="valid", name="max4")(x)

    x = Conv2D(128, (3, 3), padding='same', name="conv6")(x)

    x = BatchNormalization(name="batch6")(x)

    x = Activation('relu', name='relu6')(x)

    x = Dropout(0.4, name='dropout3')(x)

    

    x = Flatten(name='flatten')(x)

    

    x = Dense(128, name='dense1')(x)

    x = BatchNormalization(name="batch7")(x)

    x = Activation('relu', name="relu7")(x)

    x = Dropout(0.45, name="dropout4")(x)

    

    x = Dense(10, activation='softmax', name="dense2")(x)

    

    model = Model(inputs=input_layer, outputs=x)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])

    

    return model
def get_generators(data_train, data_test):

    # Set variables

    df_train = data_train.copy()

    df_test = data_test.copy()



    # Spliting data by gettig index

    train_idx, val_idx = train_test_split(

        df_train.index, random_state=42, test_size=0.1, shuffle=True

    )

    

    # Parameters for generator # Put outside of the generator

    params = {'dim': IMG_SHAPE,

              'batch_size': 32,

              'n_classes': 10,

              'n_channels': 1}

    

    # Generators train/val

    X = df_train.drop(['label'], axis=1).values

    y = df_train['label'].values

    

    training_generator = DataGenerator(data=X, 

                                       labels=y,

                                       list_IDs=train_idx,

                                       **params,

                                       mode='fit',

                                       shuffle=True, augment=False)

    validation_generator = DataGenerator(data=X,

                                         labels=y,

                                         list_IDs=val_idx,

                                         **params,

                                         mode='fit',

                                         shuffle=True, augment=False)

    

    # Generator test

    X_test = df_test.drop(['id'], axis=1).values

    test_generator = DataGenerator(data=X_test,

                                   list_IDs=df_test.index.values,

                                   **params,

                                   mode='predict',

                                   shuffle=False)

    

    return training_generator, validation_generator, test_generator
def train_model(gen_train, gen_val):

    model = build_model()



    cbs = [ReduceLROnPlateau(monitor='loss',

                             factor=0.5,

                             patience=1,

                             min_lr=1e-5,

                             verbose=0,

                             skip_mismatch=True),

           EarlyStopping(monitor='val_loss',

                         min_delta=1e-6,

                         patience=10,

                         verbose=1,

                         mode='auto',

                         restore_best_weights=True)]

    model.summary()

    history = model.fit_generator(gen_train, 

                        epochs=50,

                        validation_data=gen_val, 

                        validation_steps=len(gen_val), 

                        shuffle=True, 

                        callbacks=cbs, 

                        verbose=1)

    return model, history
train_gen, val_gen, test_gen = get_generators(data_train, data_test)

model, history = train_model(train_gen, val_gen)

y_pred = model.predict_generator(test_gen, use_multiprocessing=True)
def plot_loss(history):

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    

    # Plot train/val accuracy

    ax[0].plot(history.history['accuracy'])

    ax[0].plot(history.history['val_accuracy'])

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
def plot_lr(history):

    fig, ax = plt.subplots(figsize=(7, 5))

    

    # Plot learning rate

    ax.plot(history.history['lr'])

    ax.set_title('Learning rate evolution')

    ax.set_ylabel('Learning rate value')

    ax.set_xlabel('Epochs')

    ax.legend(['Train'], loc='upper right')
plot_lr(history)
def print_results(history):

    print("ACCURACY :")

    print(f"Training accuracy : {history.history['accuracy'][-1]}")

    print(f"Validation accuracy : {history.history['val_accuracy'][-1]}")

    

    print("\nLOSS :")

    print(f"Training categorical crossentropy loss : {history.history['loss'][-1]}")

    print(f"Validation categorical crossentropy loss : {history.history['val_loss'][-1]}")
print_results(history)
data_test_label = data_test.copy() 

y_pred_label = np.argmax(y_pred, axis=1)

data_test_label['label'] = y_pred_label
plot_images(data_test_label.drop(['id'], axis=1))
def create_submission(data):

    df = pd.DataFrame()

    df['id'] = data['id']

    df['label'] = data['label']

    return df
submission = create_submission(data_test_label)

submission.to_csv('submission.csv', index=False)