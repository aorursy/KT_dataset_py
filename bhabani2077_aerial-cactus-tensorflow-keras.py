# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import tensorflow as tf

import cv2

import seaborn as sns

import matplotlib.pyplot as plt

from PIL import Image

# Any results you write to the current directory are saved as output.
print(tf.__version__)
sns.set(font_scale=1.5)
MAIN_DIR = '../input/'

TRAIN_DIR = os.path.join(MAIN_DIR,"train/train/")

TEST_DIR = os.path.join(MAIN_DIR, "test/test/")

IMAGE_ROWS = 32

IMAGE_COLS = 32

CHANNELS = 3

BATCH_SIZE = 256

EPOCHS = 300

IMG_SHAPE = (IMAGE_ROWS, IMAGE_COLS,CHANNELS)
df_train = pd.read_csv(MAIN_DIR + "train.csv")
df_train['has_cactus'].value_counts()
df_train.isnull().any()
def cnn_graph():

    main_input = tf.keras.Input(shape=(IMAGE_ROWS,IMAGE_COLS,CHANNELS), name="main_input")

    x = tf.keras.layers.Conv2D(16, (3,3), padding='same', use_bias=False, name="conv_1")(main_input)

    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Activation('relu')(x)



    x = tf.keras.layers.Conv2D(32, (3,3), padding='same', use_bias=False, name='conv_2')(x)

    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Dropout(0.25)(x)



    x = tf.keras.layers.Conv2D(64, (3,3), padding='same', use_bias=False, name="conv_3")(x)

    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Activation('relu')(x)



    x = tf.keras.layers.Conv2D(64, (3,3), padding='same', use_bias=False, name="conv_4")(x)

    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Dropout(0.25)(x)



    x = tf.keras.layers.Conv2D(128, (3,3), padding='same', use_bias=False, name="conv_5")(x)

    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Activation('relu')(x)



    x = tf.keras.layers.Conv2D(128, (3,3), padding='same', use_bias=False, name="conv_6")(x)

    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Dropout(0.25)(x)



    x = tf.keras.layers.Conv2D(256, (3,3), padding='same', use_bias=False, name="conv_7")(x)

    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Activation('relu')(x)



    x = tf.keras.layers.Conv2D(256, (3,3), padding='same', use_bias=False, name="conv_8")(x)

    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Dropout(0.25)(x)



    x = tf.keras.layers.Conv2D(512, (3,3), padding='same', use_bias=False, name="conv_9")(x)

    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(512, (3,3), padding='same', use_bias=False, name='conv_10')(x)

    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Dropout(0.25)(x)



    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(256, activation='relu', use_bias=True)(x)

    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Dropout(0.25)(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name="output_sigmoid")(x)

    

    model = tf.keras.Model(inputs=main_input, outputs=outputs)

    

    return model
df = pd.read_csv(MAIN_DIR+'train.csv')

df['has_cactus'] = df['has_cactus'].apply(str)
model = cnn_graph()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),

                                   loss = 'binary_crossentropy',

                                   metrics=['accuracy'])
def load_data(dataframe=None, batch_size=16, class_mode='binary', target_size=(IMAGE_ROWS, IMAGE_COLS)):

    k_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,

                                                            zoom_range=0.1,

                                                           validation_split=0.1,

                                                           horizontal_flip=True,

                                                           vertical_flip=True,

                                                           fill_mode='nearest')

    data_train = k_gen.flow_from_dataframe(dataframe,

                                           directory=TRAIN_DIR,

                                           x_col='id',

                                           y_col='has_cactus',

                                           has_ext=True,

                                           target_size=target_size,

                                           class_mode=class_mode,

                                           batch_size=batch_size,

                                           shuffle=True,

                                           subset="training")

    

    data_test = k_gen.flow_from_dataframe(dataframe,

                                          directory=TRAIN_DIR,

                                          x_col='id',

                                          y_col='has_cactus',

                                          has_ext=True,

                                          target_size=(target_size),

                                          class_mode=class_mode,

                                          batch_size=batch_size,

                                          shuffle=True,

                                          subset='validation')

    

    return data_train, data_test
train_k_gen, val_k_gen = load_data(dataframe=df,batch_size=BATCH_SIZE, target_size=(IMAGE_ROWS, IMAGE_COLS))
model.summary()
history = model.fit_generator(train_k_gen, 

                              steps_per_epoch=(train_k_gen.n//train_k_gen.batch_size),

                              epochs=EPOCHS,

                              validation_data=val_k_gen,

                              validation_steps=len(val_k_gen),

                              shuffle=True)
import tqdm

from tqdm import tqdm, tqdm_notebook
def predict(model, sample_submission):

    pred = np.empty((sample_submission.shape[0],))

    for n in tqdm(range(sample_submission.shape[0])):

        data = np.array(Image.open(TEST_DIR + sample_submission.id[n]))

        pred[n] = model.predict(data.reshape((1, 32, 32, 3))/255)[0]

    

    sample_submission['has_cactus'] = pred

    return sample_submission
sample_submission = pd.read_csv(MAIN_DIR + 'sample_submission.csv')

df_prediction = predict(model, sample_submission)



df_prediction.to_csv('submission_1.csv', index=False)
df_prediction.head()