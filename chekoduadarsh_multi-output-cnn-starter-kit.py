#Load em...

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

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.models import clone_model

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import plot_model



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import numpy as np

import datetime as dt





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')
def resize(df, size=64, need_progress_bar=True):

    resized = {}

    for i in range(df.shape[0]):

        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

        resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T

    return resized
inputs = keras.Input(shape = (64, 64, 1))



model = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(inputs)

model = tf.nn.leaky_relu(model, alpha=0.01, name='Leaky_ReLU') 

model = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = tf.nn.leaky_relu(model, alpha=0.01, name='Leaky_ReLU') 

model = layers.BatchNormalization(momentum=0.15)(model)

model = layers.MaxPool2D(pool_size=(2, 2))(model)

model = layers.Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = tf.nn.leaky_relu(model, alpha=0.01, name='Leaky_ReLU') 

model = layers.Dropout(rate=0.3)(model)



model = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(inputs)

model = tf.nn.leaky_relu(model, alpha=0.01, name='Leaky_ReLU') 

model = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = tf.nn.leaky_relu(model, alpha=0.01, name='Leaky_ReLU') 

model = layers.BatchNormalization(momentum=0.15)(model)

model = layers.MaxPool2D(pool_size=(2, 2))(model)

model = layers.Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = tf.nn.leaky_relu(model, alpha=0.01, name='Leaky_ReLU') 

model = layers.Dropout(rate=0.3)(model)





model = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = tf.nn.leaky_relu(model, alpha=0.01, name='Leaky_ReLU') 

model = layers.BatchNormalization(momentum=0.15)(model)

model = layers.Dropout(rate=0.3)(model)

model = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = tf.nn.leaky_relu(model, alpha=0.01, name='Leaky_ReLU') 

model = layers.BatchNormalization(momentum=0.15)(model)

model = layers.Dropout(rate=0.3)(model)

model = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = tf.nn.leaky_relu(model, alpha=0.01, name='Leaky_ReLU') 

model = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = tf.nn.leaky_relu(model, alpha=0.01, name='Leaky_ReLU') 

model = layers.BatchNormalization(momentum=0.15)(model)

model = layers.MaxPool2D(pool_size=(2, 2))(model)

model = layers.Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = tf.nn.leaky_relu(model, alpha=0.01, name='Leaky_ReLU') 

model = layers.BatchNormalization(momentum=0.15)(model)

model = layers.Dropout(rate=0.3)(model)



model = layers.Flatten()(model)

model = layers.Dense(1024, activation = "relu")(model)

model = layers.Dropout(rate=0.3)(model)

dense = layers.Dense(512, activation = "relu")(model)

    

out_root = layers.Dense(168, activation='softmax', name='root_out')(dense) 

out_vowel = layers.Dense(11, activation='softmax', name='vowel_out')(dense) 

out_consonant = layers.Dense(7, activation='softmax', name='consonant_out')(dense)

    



cnn1model = keras.Model(inputs=inputs, outputs=[out_root, out_vowel, out_consonant])



plot_model(cnn1model, to_file='mode4.png')
cnn1model.compile(optimizer="adam", loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'], metrics=['accuracy']) # Adam optimizer with catagorical_crossentropy modern best
mc = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5', save_weights_only=True, period=10)
batch_size = 128

epochs = 20

history_list = []

paraquets_used = 2
class MultiOutputDataGenerator(keras.preprocessing.image.ImageDataGenerator):



    def flow(self,

             x,

             y=None,

             batch_size=32,

             shuffle=True,

             sample_weight=None,

             seed=None,

             save_to_dir=None,

             save_prefix='',

             save_format='png',

             subset=None):



        targets = None

        target_lengths = {}

        ordered_outputs = []

        for output, target in y.items():

            if targets is None:

                targets = target

            else:

                targets = np.concatenate((targets, target), axis=1)

            target_lengths[output] = target.shape[1]

            ordered_outputs.append(output)





        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,

                                         shuffle=shuffle):

            target_dict = {}

            i = 0

            for output in ordered_outputs:

                target_length = target_lengths[output]

                target_dict[output] = flowy[:, i: i + target_length]

                i += target_length



            yield flowx, target_dict
for i in range(paraquets_used):

    b_train_data =  pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet'), train_data, on='image_id').drop(['image_id'], axis =1)

    print("Data Loaded")

    x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split((resize(b_train_data.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic','grapheme'], axis=1))/255).values.reshape(-1, 64, 64, 1), pd.get_dummies(b_train_data['grapheme_root']).values, pd.get_dummies(b_train_data['vowel_diacritic']).values, pd.get_dummies(b_train_data['consonant_diacritic']).values, test_size=0.08, random_state=666)

    

    del b_train_data

    gc.collect()

    

    #del Y_train_root, Y_train_vowel, Y_train_consonant

    print("Del1")

    datagen = MultiOutputDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.20, # Randomly zoom image 

        width_shift_range=0.20,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.20,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





        # This will just calculate parameters required to augment the given data. This won't perform any augmentations

    datagen.fit(x_train)

    history = cnn1model.fit_generator(datagen.flow(x_train, {'root_out': y_train_root, 'vowel_out': y_train_vowel, 'consonant_out': y_train_consonant}, batch_size=batch_size),

            epochs = epochs, validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]), 

            steps_per_epoch=x_train.shape[0] // batch_size,

            callbacks=[mc])

                                  

    history_list.append(history)

    del datagen

    del x_train

    del x_test

    del y_train_root

    del y_test_root

    del y_train_vowel

    del y_test_vowel

    del y_train_consonant

    del y_test_consonant    

    print("Del2")

    gc.collect()

del train_data

gc.collect()
%matplotlib inline

def plot_loss(his, epoch, title):

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')

    plt.plot(np.arange(0, epoch), his.history['root_out_loss'], label='train_root_loss')

    plt.plot(np.arange(0, epoch), his.history['vowel_out_loss'], label='train_vowel_loss')

    plt.plot(np.arange(0, epoch), his.history['consonant_out_loss'], label='train_consonant_loss')

    

    plt.plot(np.arange(0, epoch), his.history['val_root_out_loss'], label='val_train_root_loss')

    plt.plot(np.arange(0, epoch), his.history['val_vowel_out_loss'], label='val_train_vowel_loss')

    plt.plot(np.arange(0, epoch), his.history['val_consonant_out_loss'], label='val_train_consonant_loss')

    

    plt.title(title)

    plt.xlabel('Epoch #')

    plt.ylabel('Loss')

    plt.legend(loc='upper right')

    plt.show()



def plot_acc(his, epoch, title):

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history['root_out_accuracy'], label='train_root_acc')

    plt.plot(np.arange(0, epoch), his.history['vowel_out_accuracy'], label='train_vowel_accuracy')

    plt.plot(np.arange(0, epoch), his.history['consonant_out_accuracy'], label='train_consonant_accuracy')

    

    plt.plot(np.arange(0, epoch), his.history['val_root_out_accuracy'], label='val_root_acc')

    plt.plot(np.arange(0, epoch), his.history['val_vowel_out_accuracy'], label='val_vowel_accuracy')

    plt.plot(np.arange(0, epoch), his.history['val_consonant_out_accuracy'], label='val_consonant_accuracy')

    plt.title(title)

    plt.xlabel('Epoch #')

    plt.ylabel('Accuracy')

    plt.legend(loc='upper right')

    plt.show()
for dataset in range(paraquets_used):

    plot_loss(history_list[dataset], epochs, f'Training Dataset: {dataset}')

    plot_acc(history_list[dataset], epochs, f'Training Dataset: {dataset}')

preds_dict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}
components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target=[] # model predictions placeholder

row_id=[] # row_id place holder

for i in range(4):

    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

    df_test_img.set_index('image_id', inplace=True)



    X_test = resize(df_test_img, need_progress_bar=False)/255

    X_test = X_test.values.reshape(-1, 64, 64, 1)

    

    preds = cnn1model.predict(X_test)



    for i, p in enumerate(preds_dict):

        preds_dict[p] = np.argmax(preds[i], axis=1)



    for k,id in enumerate(df_test_img.index.values):  

        for i,comp in enumerate(components):

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
