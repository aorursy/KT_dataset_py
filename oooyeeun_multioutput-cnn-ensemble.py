# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm.auto import tqdm

from glob import glob

import time, gc

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def resize(df, size=64, need_progress_bar=True):

    resized = {}

    resize_size=64

    if need_progress_bar:

        for i in tqdm(range(df.shape[0])):

            image=df.loc[df.index[i]].values.reshape(137,236)



            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]



            idx = 0 

            ls_xmin = []

            ls_ymin = []

            ls_xmax = []

            ls_ymax = []

            for cnt in contours:

                idx += 1

                x,y,w,h = cv2.boundingRect(cnt)

                ls_xmin.append(x)

                ls_ymin.append(y)

                ls_xmax.append(x + w)

                ls_ymax.append(y + h)

            xmin = min(ls_xmin)

            ymin = min(ls_ymin)

            xmax = max(ls_xmax)

            ymax = max(ls_ymax)



            roi = image[ymin:ymax,xmin:xmax]

            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)

            resized[df.index[i]] = resized_roi.reshape(-1)

    else:

        for i in range(df.shape[0]):

            #image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size),None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)

            image=df.loc[df.index[i]].values.reshape(137,236)





            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]



            idx = 0 

            ls_xmin = []

            ls_ymin = []

            ls_xmax = []

            ls_ymax = []

            for cnt in contours:

                idx += 1

                x,y,w,h = cv2.boundingRect(cnt)

                ls_xmin.append(x)

                ls_ymin.append(y)

                ls_xmax.append(x + w)

                ls_ymax.append(y + h)

            xmin = min(ls_xmin)

            ymin = min(ls_ymin)

            xmax = max(ls_xmax)

            ymax = max(ls_ymax)



            roi = image[ymin:ymax,xmin:xmax]

            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)

            resized[df.index[i]] = resized_roi.reshape(-1)

    resized = pd.DataFrame(resized).T

    return resized
IMG_SIZE=64

N_CHANNELS=1



batch_size = 256

epochs = 20



HEIGHT = 137

WIDTH = 236
from keras.models import Model, clone_model, Sequential

from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, Input

from keras.layers import AveragePooling2D, GlobalAveragePooling2D

from keras.applications.resnet50 import ResNet50

from keras.layers.advanced_activations import LeakyReLU

from keras.callbacks import ReduceLROnPlateau

from keras.optimizers import RMSprop

from keras.regularizers import l2

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns

import keras.backend as K

%matplotlib inline
def cnn_branching(i):

    inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 1))

    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu',

                   kernel_regularizer = l2(0.001))(inputs)

    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

    model = BatchNormalization(momentum=0.15)(model)

    model = MaxPool2D(pool_size=(2, 2))(model)

    model = Dropout(rate=0.3)(model)

    

    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu',

                   kernel_regularizer = l2(0.001))(model)

    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

    model = BatchNormalization(momentum=0.15)(model)

    model = MaxPool2D(pool_size=(2, 2))(model)

    model = Dropout(rate=0.3)(model)

    

    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu',

                   kernel_regularizer = l2(0.001))(model)

    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

    model = BatchNormalization(momentum=0.15)(model)

    model = MaxPool2D(pool_size=(2, 2))(model)

    model = Dropout(rate=0.3)(model)

    

    _x = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu',

                kernel_regularizer = l2(0.001))(model)

    _x = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(_x)

    _x = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(_x)

    _x = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(_x)

    _x = BatchNormalization(momentum=0.15)(_x)

    _x = MaxPool2D(pool_size=(2, 2))(_x)

    _x = Dropout(rate=0.3)(_x)



    _x = Flatten()(_x)

    # _x = Dense(1024, activation = "relu")(_x)

    _x = Dropout(rate=0.3)(_x)

    _x = Dense(512, activation = "relu")(_x)

    root = Dense(168, activation = 'softmax', name = 'root_'+str(i))(_x)

    

    #for vowel and consonant

    _x1 = Flatten()(model)

    _x1 = Dense(256, activation = 'relu')(_x1)

    _x1 = Dropout(rate=0.3)(_x1)

    dense_1 = Dense(256, activation = 'relu')(_x1)

    vowel = Dense(11, activation='softmax', name = 'vowel_'+str(i))(dense_1)

    consonant = Dense(7, activation='softmax', name = 'consonant_'+str(i))(dense_1)



    model = Model(inputs=inputs, outputs=[root, vowel, consonant])

    model.compile(optimizer = 'adam',

              loss='categorical_crossentropy',

              loss_weights = {'root_'+str(i): 2, 'vowel_'+str(i):1, 'consonant_'+str(i):1},

              metrics=['accuracy'])

    return model
model = cnn_branching(0)

model.summary()
from tensorflow import keras

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
train_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

test_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

train_df_ = train_df_.drop(['grapheme'], axis=1, inplace=False)

train_df_[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] = train_df_[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].astype('uint8')
preds_dict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}
# Set a learning rate annealer. Learning rate will be half after 3 epochs if accuracy is not increased

def lr_reduction(i):

    learning_rate_reduction_root = ReduceLROnPlateau(monitor='root_'+str(i)+'_accuracy', 

                                                patience=3, 

                                                verbose=1,

                                                factor=0.5, 

                                                min_lr=0.000001)

    learning_rate_reduction_vowel = ReduceLROnPlateau(monitor='vowel_'+str(i)+'_accuracy', 

                                                patience=3, 

                                                verbose=1,

                                                factor=0.5, 

                                                min_lr=0.000001)

    learning_rate_reduction_consonant = ReduceLROnPlateau(monitor='vowel_'+str(i)+'_accuracy', 

                                                patience=3, 

                                                verbose=1,

                                                factor=0.5, 

                                                min_lr=0.000001)

    return learning_rate_reduction_root, learning_rate_reduction_vowel, learning_rate_reduction_consonant
gc.collect()

nets = 2

histories = []

models = []

model_1 = cnn_branching(1)

model_2 = cnn_branching(2)

# model_3 = cnn_branching(3)

root_1, vowel_1, cons_1 = lr_reduction(1)

root_2, vowel_2, cons_2 = lr_reduction(2)

# root_3, vowel_3, cons_3 = lr_reduction(3)





for i in range(4):

    hist = []

    print("######################training_loop_{}################################".format(i))

    train_df = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)

    

    

    X_train = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)

    X_train = resize(X_train)/255

    

    # CNN takes images in shape `(batch_size, h, w, channels)`, so reshape the images

    X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

    

    Y_train_root = pd.get_dummies(train_df['grapheme_root']).values

    Y_train_vowel = pd.get_dummies(train_df['vowel_diacritic']).values

    Y_train_consonant = pd.get_dummies(train_df['consonant_diacritic']).values



    print(f'Training images: {X_train.shape}')

    print(f'Training labels root: {Y_train_root.shape}')

    print(f'Training labels vowel: {Y_train_vowel.shape}')

    print(f'Training labels consonants: {Y_train_consonant.shape}')



    # Divide the data into training and validation set

    x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.08, random_state=666)

    del train_df

    del X_train

    del Y_train_root, Y_train_vowel, Y_train_consonant



    # Data augmentation for creating more training data

    datagen = MultiOutputDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.15, # Randomly zoom image 

        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





    # This will just calculate parameters required to augment the given data. This won't perform any augmentations

    datagen.fit(x_train)



    # Fit the model

    print("CNN----------------------------1")

    hist_1 = model_1.fit_generator(datagen.flow(x_train, {'root_1': y_train_root, 'vowel_1': y_train_vowel, 'consonant_1': y_train_consonant}, batch_size=batch_size),

                            epochs = epochs, validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]), 

                            steps_per_epoch=x_train.shape[0] // batch_size, 

                            #   class_weight = class_weights,

                            callbacks=[root_1, vowel_1, cons_1]

                            )

    print("CNN----------------------------2")

    hist_2 = model_2.fit_generator(datagen.flow(x_train, {'root_2': y_train_root, 'vowel_2': y_train_vowel, 'consonant_2': y_train_consonant}, batch_size=batch_size),

                        epochs = epochs, validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]), 

                        steps_per_epoch=x_train.shape[0] // batch_size, 

                        #   class_weight = class_weights,

                        callbacks=[root_2, vowel_2, cons_2]

                        )

#     print("CNN----------------------------3")

#     hist_3 = model_3.fit_generator(datagen.flow(x_train, {'root_3': y_train_root, 'vowel_3': y_train_vowel, 'consonant_3': y_train_consonant}, batch_size=batch_size),

#                         epochs = epochs, validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]), 

#                         steps_per_epoch=x_train.shape[0] // batch_size, 

#                         #   class_weight = class_weights,

#                         callbacks=[root_3, vowel_3, cons_3]

#                         )



    hist.append(hist_1)

    hist.append(hist_2)

#     hist.append(hist_3)

    histories.append(hist)

    del hist_1

    del hist_2

#     del hist_3

    # Delete to reduce memory usage

    del x_train

    del x_test

    del y_train_root

    del y_test_root

    del y_train_vowel

    del y_test_vowel

    del y_train_consonant

    del y_test_consonant

def plot_loss(his, epoch, title):

    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')

    plt.plot(np.arange(0, epoch), his.history['dense_3_loss'], label='train_root_loss')

    plt.plot(np.arange(0, epoch), his.history['dense_6_loss'], label='train_vowel_loss')

    plt.plot(np.arange(0, epoch), his.history['dense_7_loss'], label='train_consonant_loss')

    

    plt.plot(np.arange(0, epoch), his.history['val_dense_3_loss'], label='val_train_root_loss')

    plt.plot(np.arange(0, epoch), his.history['val_dense_6_loss'], label='val_train_vowel_loss')

    plt.plot(np.arange(0, epoch), his.history['val_dense_7_loss'], label='val_train_consonant_loss')

    

    plt.title(title)

    plt.xlabel('Epoch #')

    plt.ylabel('Loss')

    plt.legend(loc='upper right')

    plt.show()



def plot_acc(his, epoch, title):

    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history['dense_3_accuracy'], label='train_root_accuracy')

    plt.plot(np.arange(0, epoch), his.history['dense_6_accuracy'], label='train_vowel_accuracy')

    plt.plot(np.arange(0, epoch), his.history['dense_7_accuracy'], label='train_consonant_accuracy')

    

    plt.plot(np.arange(0, epoch), his.history['val_dense_3_accuracy'], label='val_root_accuracy')

    plt.plot(np.arange(0, epoch), his.history['val_dense_6_accuracy'], label='val_vowel_accuracy')

    plt.plot(np.arange(0, epoch), his.history['val_dense_7_accuracy'], label='val_consonant_accuracy')

    plt.title(title)

    plt.xlabel('Epoch #')

    plt.ylabel('Accuracy')

    plt.legend(loc='upper right')

    plt.show()
# for dataset in range(4):

#     for i in range(nets):

#         plot_loss(histories[dataset][i], epochs, f'Training Dataset: {dataset},'+f"model_{i}")

#         plot_acc(histories[dataset][i], epochs, f'Training Dataset: {dataset},'+f"model_{i}")
del histories

gc.collect()
# preds_dict = {

#     'grapheme_root': [],

#     'vowel_diacritic': [],

#     'consonant_diacritic': []

# }
preds_dict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}

components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target=[] # model predictions placeholder

row_id=[] # row_id place holder



#nets: 모델개수 models list에 각 모델 넣어주기

# nets = 2

models = [0] * nets

models[0] = model_1

models[1] = model_2

# models[2] = model_3





for i in range(4):

    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

    df_test_img.set_index('image_id', inplace=True)



    X_test = resize(df_test_img, need_progress_bar=False)/255

    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)



    results_roots = np.zeros((X_test.shape[0], 168))

    results_vowels = np.zeros((X_test.shape[0], 11))

    results_consonants = np.zeros((X_test.shape[0], 7))

    results_roots = results_roots.astype('float')

    results_vowels = results_vowels.astype('float')

    results_consonants = results_consonants.astype('float')



    for i in range(nets):

        results_roots += models[i].predict(X_test)[0]

        results_vowels += models[i].predict(X_test)[1]

        results_consonants += models[i].predict(X_test)[2]

      #preds[0] = results_roots

      #preds[1] = results_vowels

      #preds[2] = results_consonants



    results_roots = np.argmax(results_roots, axis = 1)

    results_vowels = np.argmax(results_vowels, axis = 1)

    results_consonants = np.argmax(results_consonants, axis = 1)



    preds_dict['grapheme_root'] = results_roots

    preds_dict['vowel_diacritic'] = results_vowels

    preds_dict['consonant_diacritic'] = results_consonants



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