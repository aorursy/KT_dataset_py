#Load libraries

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont

import random

import os

import cv2

import gc

from tqdm.auto import tqdm

import sys

import random



import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras import regularizers

from tensorflow.keras.utils import plot_model  

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# NN paramz:

network_deepth = 5



# train paramz:

epochs = 3

traintestsplit = 0

batch_size = 88

shape_base = (236, 137)

shape_scale_fuctor = 3 # is used to calcualte shape for NN input layer

optimizer = 'adam'



# where infomation is stored:

input_dir = '/kaggle/input/bengaliai-cv19/'



# compile shape:

shape = (shape_base[0] / shape_scale_fuctor, shape_base[1] / shape_scale_fuctor)

shape = (int(shape[0]), int(shape[1]))

print('image size: %i x %i' % (shape[0], shape[1])) 
# loading train data and optimizing for memory economy

print("data loading")

train_data  = pd.read_csv(input_dir + 'train.csv')

train_data['grapheme_root'] = train_data['grapheme_root'].astype('uint8')

train_data['vowel_diacritic'] = train_data['vowel_diacritic'].astype('uint8')

train_data['consonant_diacritic'] = train_data['consonant_diacritic'].astype('uint8')



train_data.describe()
def resize(df, shape):

    resized_dic = {}

    for i in tqdm(range(df.shape[0])):

        resized_dic[df.index[i]] = cv2.resize(df.loc[df.index[i]].values.reshape(shape_base[1],shape_base[0]),shape, interpolation = cv2.INTER_LINEAR).reshape(-1).astype(np.float32) / 255

        if i%500 == 0: # memory clearing

            gc.collect()

    resized = pd.DataFrame(resized_dic).T

    del resized_dic

    gc.collect()

    return resized
def res_net_block_1(input_data, filters):

    x1 = layers.Conv2D(filters, 3, activation='relu', padding='same')(input_data)

    x1 = layers.LeakyReLU(alpha=0.01)(x1)

    x2 = layers.BatchNormalization()(x1)

    x2 = layers.Dropout(0.1)(x2)



    x3 = layers.Conv2D(filters, 5, activation=None, padding='same')(x2)

    x3 = layers.LeakyReLU(alpha=0.01)(x3)

    x4 = layers.BatchNormalization()(x3)

    x4 = layers.Dropout(0.1)(x4)



    x5 = layers.Conv2D(filters, 1, activation=None, padding='same')(input_data)

    x5 = layers.LeakyReLU(alpha=0.01)(x5)



    x = layers.Add()([x4, x5])

    x = layers.Activation('relu')(x)

    return x



def res_net_block_2(input_data, filters):

    x1 = layers.Conv2D(filters, 3, activation='relu', padding='same')(input_data)

    x1 = layers.LeakyReLU(alpha=0.01)(x1)

    x2 = layers.BatchNormalization()(x1)

    x2 = layers.Dropout(0.1)(x2)



    x3 = layers.Conv2D(filters, 5, activation=None, padding='same')(input_data)

    x3 = layers.LeakyReLU(alpha=0.01)(x3)

    x4 = layers.BatchNormalization()(x3)

    x4 = layers.Dropout(0.1)(x4)



    x5 = layers.Conv2D(filters, 1, activation=None, padding='same')(input_data)

    x5 = layers.LeakyReLU(alpha=0.01)(x5)



    x = layers.Add()([x2, x4, x5])

    x = layers.Activation('relu')(x)

    return x



# multy output

def resnet_multiOutput(input_shape, outputsizes, num_res_net_blocks):

    inputs = layers.Input(shape=(input_shape[1],input_shape[0],1))

    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)

    x = layers.LeakyReLU(alpha=0.01, name='Leaky_ReLU_1')(x)

    x = layers.Conv2D(64, (3,3), activation='relu')(x)

    x = layers.LeakyReLU(alpha=0.01, name='Leaky_ReLU_2')(x)

    x = layers.MaxPooling2D(3)(x)

    x = layers.Dropout(0.1)(x)



    for i in range(num_res_net_blocks):

        x = res_net_block_1(x, 64)

        x = res_net_block_2(x, 64)

        

    x = layers.Conv2D(64, 3, activation='relu')(x)

    x = layers.LeakyReLU(alpha=0.01, name='Leaky_ReLU_3')(x)

    x = layers.GlobalAveragePooling2D()(x)

    

    # dence layers

    dense = layers.Dense(1024, activation='relu')(x)

    dense = layers.Dropout(0.5)(dense)

    dense = layers.Dense(512, activation='relu')(x)

    dense = layers.Dropout(0.5)(dense)

    

    # output layers

    head_root = layers.Dense(outputsizes[0], activation = 'softmax', name='dense_grapheme_root')(dense)

    head_vowel = layers.Dense(outputsizes[1], activation = 'softmax', name='dense_vowel_diacritic')(dense)

    head_consonant = layers.Dense(outputsizes[2], activation = 'softmax', name='dense_consonant_diacritic')(dense)

    

    model = tf.keras.Model(inputs, [head_root, head_vowel, head_consonant])

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
class MultiOutputDataGenerator(ImageDataGenerator):

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
histories = []

def trainMultiOutput(ds_num, batch_size, epochs, model):

    print('loading dataset %i' %  ds_num)

    b_train_data = pd.merge(pd.read_parquet(input_dir + f'train_image_data_{ds_num}.parquet'), train_data, on='image_id').drop(['image_id'], axis=1)

    gc.collect()

    train_image = resize(b_train_data.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme'], axis=1), shape)

    train_image = train_image.values.reshape(-1, shape[1], shape[0], 1)

    gc.collect()

    

    datagen = MultiOutputDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset

                            samplewise_center=False,  # set each sample mean to 0

                            featurewise_std_normalization=False,  # divide inputs by std of the dataset

                            samplewise_std_normalization=False,  # divide each input by its std

                            zca_whitening=False,  # apply ZCA whitening

                            rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)

                            zoom_range=0.15,  # Randomly zoom image

                            width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)

                            height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)

                            horizontal_flip=False,  # randomly flip images

                            vertical_flip=False)  # randomly flip images



    # This will just calculate parameters required to augment the given data. This won't perform any augmentations

    datagen.fit(train_image)



    # traintest split

    x_train = train_image

    y_train_root = pd.get_dummies(b_train_data['grapheme_root']).values

    y_train_vowel = pd.get_dummies(b_train_data['vowel_diacritic']).values

    y_train_consonant = pd.get_dummies(b_train_data['consonant_diacritic']).values

    

    if traintestsplit > 0:

        x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(

            train_image, y_train_root, y_train_vowel, y_train_consonant, test_size=traintestsplit, random_state=999)

        del train_image

        del b_train_data

    

    # fit

    gc.collect()

    history = model.fit_generator(datagen.flow(x_train, {'dense_grapheme_root': y_train_root, 'dense_vowel_diacritic': y_train_vowel, 'dense_consonant_diacritic': y_train_consonant},

                                                   batch_size=batch_size),

                                                   epochs=epochs, validation_data=(x_test, [y_test_root, y_test_vowel, y_test_consonant]) if traintestsplit > 0 else None,

                                                   steps_per_epoch = x_train.shape[0] // batch_size,

                                                   #callbacks=[learning_rate_reduction], 

                                                   verbose=2)

    histories.append(history)

    

    del datagen

    del y_train_root

    del y_train_vowel

    del y_train_consonant



    if traintestsplit == 0:

        del train_image

        del b_train_data

    else:

        del x_train

        del x_test

        del y_test_root

        del y_test_vowel

        del y_test_consonant

        

    gc.collect()

    print('trained')
# TRAIN HERE

model = resnet_multiOutput(shape, [168, 11, 7], network_deepth)



# fit

trainMultiOutput(0, batch_size, epochs, model)

trainMultiOutput(1, batch_size, epochs, model)

trainMultiOutput(2, batch_size, epochs, model)

trainMultiOutput(3, batch_size, epochs, model)
%matplotlib inline

def plot_loss(his, epoch, title):

    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')

    plt.plot(np.arange(0, epoch), his.history['dense_grapheme_root_loss'], label='train_root_loss')

    plt.plot(np.arange(0, epoch), his.history['dense_vowel_diacritic_loss'], label='train_vowel_loss')

    plt.plot(np.arange(0, epoch), his.history['dense_consonant_diacritic_loss'], label='train_consonant_loss')



    plt.plot(np.arange(0, epoch), his.history['dense_grapheme_root_loss'], label='val_train_root_loss')

    plt.plot(np.arange(0, epoch), his.history['dense_vowel_diacritic_loss'], label='val_train_vowel_loss')

    plt.plot(np.arange(0, epoch), his.history['dense_consonant_diacritic_loss'], label='val_train_consonant_loss')



    plt.title(title)

    plt.xlabel('Epoch #')

    plt.ylabel('Loss')

    plt.legend(loc='upper right')

    plt.show()





def plot_acc(his, epoch, title):

    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history['dense_grapheme_root_accuracy'], label='train_root_accuracy')

    plt.plot(np.arange(0, epoch), his.history['dense_vowel_diacritic_accuracy'], label='train_vowel_accuracy')

    plt.plot(np.arange(0, epoch), his.history['dense_consonant_diacritic_accuracy'], label='train_consonant_accuracy')



    plt.plot(np.arange(0, epoch), his.history['dense_grapheme_root_accuracy'], label='val_root_acc')

    plt.plot(np.arange(0, epoch), his.history['dense_vowel_diacritic_accuracy'], label='val_vowel_accuracy')

    plt.plot(np.arange(0, epoch), his.history['dense_consonant_diacritic_accuracy'], label='val_consonant_accuracy')

    plt.title(title)

    plt.xlabel('Epoch #')

    plt.ylabel('Accuracy')

    plt.legend(loc='upper right')

    plt.show()
for dataset in range(len(histories)):

    plot_loss(histories[dataset], epochs, f'Training Dataset: {dataset}')

    plot_acc(histories[dataset], epochs, f'Training Dataset: {dataset}')
# load test data

test_data = pd.read_csv(input_dir + 'test.csv')

class_map_df = pd.read_csv(input_dir + 'class_map.csv')

sample_sub_data = pd.read_csv(input_dir + 'sample_submission.csv')
perdict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}



components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target=[] # model predictions placeholder

row_id=[] # row_id place holder

n_cls = [7,168,11] # number of classes in each of the 3 targets

for i in range(4):

    print('загружаем тестовые данные %i' % i)

    df_test_img = pd.read_parquet(input_dir + 'test_image_data_{}.parquet'.format(i)) 

    df_test_img.set_index('image_id', inplace=True)



    X_test = resize(df_test_img, shape)

    X_test = X_test.values.reshape(-1, shape[1], shape[0], 1)



    preds = model.predict(X_test)

    for i, p in enumerate(perdict):

        perdict[p] = np.argmax(preds[i], axis=1)



    for k,id in enumerate(df_test_img.index.values):  

        for i,comp in enumerate(components):

            id_sample=id+'_'+comp

            row_id.append(id_sample)

            target.append(perdict[comp][k])



df_sample = pd.DataFrame(

    {'row_id': row_id,

    'target':target

    },

    columns =['row_id','target'] 

)

df_sample.to_csv('submission.csv',index=False)

print('submission saved !!!')

gc.collect()
df_sample.head(20)