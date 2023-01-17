from matplotlib import pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont

import random 

import os

import cv2

import gc

from tqdm.auto import tqdm





import numpy as np

import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.models import clone_model

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import plot_model



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import numpy as np

import datetime as dt

now = dt.datetime.now





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
num_classes = {

    'grapheme_root': 168,

    'vowel_diacritic': 11,

    'consonant_diacritic': 7

}
def resize(df, size=64, need_progress_bar=True):

    resized = {}

    for i in range(df.shape[0]):

        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

        resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T

    return resized
train_data = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')



train_data =  pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_0.parquet'), train_data, on='image_id').drop(['image_id'], axis=1)



train_labels = train_data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic','grapheme']]



train_data = train_data.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic','grapheme'], axis=1)



train_data = resize(train_data)/255



train_data = train_data.values.reshape(-1, 64, 64, 1)
def train_model(model, train_data, train_labels, target, batch_size, epochs):

    labels = train_labels[target]

    labels = pd.get_dummies(labels).values

    x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.1, random_state=123)

    y_train = tf.cast(y_train, tf.int32)

    y_test = tf.cast(y_test, tf.int32)

    print('x_train shape:', x_train.shape)

    print(x_train.shape[0], 'train samples')

    print(x_test.shape[0], 'test samples')



    model.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])



    datagen = ImageDataGenerator()

    checkpoint_cb = ModelCheckpoint("{}.h5".format(target), save_best_only=True)

    early_stopping_cb = EarlyStopping(patience=3, restore_best_weights=True)

    t = now()

    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), 

                                               epochs = epochs, validation_data = (x_test, y_test), callbacks=[checkpoint_cb, early_stopping_cb])

    

    model = keras.models.load_model("{}.h5".format(target))

    print('Training time: %s' % (now() - t))

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test score:', score[0])

    print('Test accuracy:', score[1])
feature_layers = [

    Conv2D(32, 5, activation="relu", padding="same", input_shape=[64, 64, 1]),

    layers.BatchNormalization(momentum=0.15),

    MaxPooling2D(2),

    Conv2D(64, 3, activation="relu", padding="same"),

    Conv2D(64, 3, activation="relu", padding="same"),

    MaxPooling2D(2),

    Conv2D(128, 3, activation="relu", padding="same"),

    Conv2D(128, 3, activation="relu", padding="same"),

    MaxPooling2D(2),

    Conv2D(256, 3, activation="relu", padding="same"),

    Conv2D(256, 3, activation="relu", padding="same"),

    MaxPooling2D(2),

    Flatten(),

]



classification_layers = [

    Dense(1024),

    Activation('relu'),

    Dropout(0.5),

    Dense(1024),

    Activation('relu'),

    Dropout(0.5),

    Dense(num_classes['grapheme_root']),

    Activation('softmax')

]
model_grapheme_root = Sequential(feature_layers + classification_layers)
batch_size = 32

epochs = 10
train_model(model_grapheme_root, train_data, train_labels, 'grapheme_root', batch_size, epochs)
classification_layers = [

    Dense(1024),

    Activation('relu'),

    Dropout(0.5),

    Dense(512),

    Activation('relu'),

    Dropout(0.5),

    Dense(num_classes['vowel_diacritic']),

    Activation('softmax')

]
# freeze feature layers and rebuild model

for l in feature_layers:

    l.trainable = False



model_vowel_diacritic = Sequential(feature_layers + classification_layers)



# transfer: train dense layers for new classification task

train_model(model_vowel_diacritic, train_data, train_labels, 'vowel_diacritic', batch_size, epochs)
model_dict = {

    'grapheme_root': Sequential(),

    'vowel_diacritic': Sequential(),

    'consonant_diacritic': Sequential()

}



for model_type, model in model_dict.items():

    model.add(Conv2D(32, 5, activation="relu", padding="same", input_shape=[64, 64, 1]))

    model.add(layers.BatchNormalization(momentum=0.15))

    model.add(MaxPooling2D(2))

    model.add(Conv2D(64, 3, activation="relu", padding="same"))

    model.add(Conv2D(64, 3, activation="relu", padding="same"))

    model.add(MaxPooling2D(2))

    model.add(Conv2D(128, 3, activation="relu", padding="same"))

    model.add(Conv2D(128, 3, activation="relu", padding="same"))

    model.add(MaxPooling2D(2))

    model.add(Conv2D(256, 3, activation="relu", padding="same"))

    model.add(Conv2D(256, 3, activation="relu", padding="same"))

    model.add(MaxPooling2D(2))

    model.add(Flatten())

    model.add(Dense(1024, activation="relu"))

    model.add(Dropout(0.5))

    model.add(Dense(512, activation="relu"))

    model.add(Dropout(0.5))

    if model_type == 'grapheme_root':

        model.add(layers.Dense(168, activation='softmax', name='root_out'))

    elif model_type == 'vowel_diacritic':

        model.add(layers.Dense(11, activation='softmax', name='vowel_out'))

    elif model_type == 'consonant_diacritic':

        model.add(layers.Dense(7, activation='softmax', name='consonant_out'))



    model.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])
# del Y_train

# del x_train

# del x_test

# del y_train

# del y_test

# gc.collect()
keras.backend.clear_session()

np.random.seed(42)

tf.random.set_seed(42)

model_types = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']

for target in model_types:

    Y_train = train_labels[target]

    Y_train = pd.get_dummies(Y_train).values

    x_train, x_test, y_train, y_test = train_test_split(train_data, Y_train, test_size=0.1, random_state=123)

    y_train = tf.cast(y_train, tf.int32)

    y_test = tf.cast(y_test, tf.int32)

#     datagen = ImageDataGenerator(rotation_range=5)

    datagen = ImageDataGenerator()

    checkpoint_cb = ModelCheckpoint("{}.h5".format(target), save_best_only=True)

    early_stopping_cb = EarlyStopping(patience=3, restore_best_weights=True)

    history = model_dict[target].fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), 

                                               epochs = epochs, validation_data = (x_test, y_test), callbacks=[checkpoint_cb, early_stopping_cb])

    history_list.append(history)

    model_dict[target] = keras.models.load_model("{}.h5".format(target))

    

    del Y_train

    del x_train

    del x_test

    del y_train

    del y_test

    gc.collect()
plt.figure()

for i in range(3):

    plt.plot(np.arange(0, len(history_list[i].history['accuracy'])), history_list[i].history['accuracy'], label='train_accuracy')

    plt.plot(np.arange(0, len(history_list[i].history['accuracy'])), history_list[i].history['val_accuracy'], label='val_accuracy')

    plt.title(model_types[i])

    plt.xlabel('Epoch #')

    plt.ylabel('Accuracy')

    plt.legend(loc='lower right')

    plt.show()
preds_dict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}
target=[] # model predictions placeholder

row_id=[] # row_id place holder

for i in range(4):

    print("Parquet: {}".format(i))

    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

    df_test_img.set_index('image_id', inplace=True)



    X_test = resize(df_test_img, need_progress_bar=False)/255

    X_test = X_test.values.reshape(-1, 64, 64, 1)



    for i, p in preds_dict.items():

        model = keras.models.load_model("{}.h5".format(i))

        preds = model.predict(X_test)

        preds_dict[i] = np.argmax(preds, axis=1)



    for k,id in enumerate(df_test_img.index.values):  

        for i,comp in enumerate(model_types):

            id_sample=id+'_'+comp

            row_id.append(id_sample)

            target.append(preds_dict[comp][k])

            

    del df_test_img

    del X_test

    gc.collect()



df_sample = pd.DataFrame(

    {

        'row_id': row_id,

        'target':target

    },

    columns = ['row_id','target'] 

)

df_sample.to_csv('submission.csv',index=False)

df_sample.head()