# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from tqdm.auto import tqdm

from glob import glob

import time, gc

import cv2



from tensorflow import keras

from keras import layers

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from keras.models import Model, load_model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

from keras.callbacks import ReduceLROnPlateau

import pydot

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

from keras.initializers import glorot_uniform

import scipy.misc

from matplotlib.pyplot import imshow

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

%matplotlib inline



import keras.backend as K

K.set_image_data_format('channels_last')

K.set_learning_phase(1)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load the data

train_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

test_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

sample_sub_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
# Explore the size of loaded DataFrames

print(f'Size of training data: {train_df_.shape}')

print(f'Size of test data: {test_df_.shape}')

print(f'Size of class map: {class_map_df.shape}')
# Create helpful functions for data processing

def resize(df, size=64, need_progress_bar=True):

    """Function which resizes the images to 64x64 pixels

    

    ARGS :

    - df : Data frame containing images' pixels values

    - size : size of target image (64 pixels by default)

    - need_progress_bar : display progress bar (True by default)

    

    OUTPUT:

    - dataframe of resized images

    

    Source kernel: Bengali Graphemes_Multi Output ResNet-50  

    """

    resized = {}

    if need_progress_bar:

        for i in tqdm(range(df.shape[0])):

            image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

            resized[df.index[i]] = image.reshape(-1)

    else:

        for i in range(df.shape[0]):

            image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

            resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T

    return resized



def get_dummies(df):

    """     

    Source kernel: Bengali Graphemes_Multi Output ResNet-50 """

    cols = []

    for col in df:

        cols.append(pd.get_dummies(df[col].astype(str)))

    return pd.concat(cols, axis=1)
# Delete the 'grapheme' column which is not useful for further modeling. Change the type of features to uint8.

train_df_ = train_df_.drop(['grapheme'], axis=1, inplace=False)

train_df_[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] = train_df_[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].astype('uint8')
def identity_block(X, f, filters, stage, block):

    """

    ResNet Identity block



    Arguments:

    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    f -- integer, specifying the shape the CONV's window for the main path

    filters -- an integer defining the number of filters in the CONV layers of the main path

    stage -- integer, used to name the layers, depending on their position in the network

    block -- string/character, used to name the layers, depending on their position in the network



    Returns:

    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)

    

    Source kernel : Bengali Graphemes_ Multi Output ResNet-50. 

    The block was modified to correspond to ResNet18 Architecture.

    """



    # defining name basis

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'



    # Retrieve Filter 

       

    F = filters



    # Save the input value. You'll need this later to add back to the main path. 

    X_shortcut = X



    # Component of main path

    X = Conv2D(filters=F, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)

    X = Activation('relu')(X)



    # Add shortcut value to main path, and pass it through a RELU activation

    X = Add()([X, X_shortcut])

    X = Activation('relu')(X)



    return X
def convolutional_block(X, f, filters, stage, block, s=2):

    """

    Implementation of the convolutional block



    Arguments:

    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    f -- an integer specifying the shape of the middle CONV's window for the main path

    filters -- an integer defining the number of filters in the CONV layers of the main path

    stage -- integer, used to name the layers, depending on their position in the network

    block -- string/character, used to name the layers, depending on their position in the network

    s -- Integer, specifying the stride to be used



    Returns:

    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)

    

    Source kernel : Bengali Graphemes_ Multi Output ResNet-50. 

    The block was modified to correspond to ResNet18 Architecture.

    """



    # defining name basis

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'



    # Retrieve Filters     

    F = filters



    # Save the input value

    X_shortcut = X



    # Second component of main path

    X = Conv2D(filters=F, kernel_size=(f, f), strides=(s, s), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)

    X = Activation('relu')(X)



    ##### SHORTCUT PATH #### 

    X_shortcut = Conv2D(filters=F, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)

    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)



    # Final step: Add shortcut value to main path, and pass it through a RELU activation

    X = Add()([X, X_shortcut])

    X = Activation('relu')(X)



    return X
def ResNet18(input_shape=(64, 64, 1)):

    """

    Implementation of the popular ResNet50 the following architecture:

    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK1 -> IDBLOCK1 -> CONVBLOCK2 -> IDBLOCK2

    -> CONVBLOCK3 -> IDBLOCK3 -> CONVBLOCK4 -> IDBLOCK4 -> AVGPOOL -> TOPLAYERS



    Arguments:

    input_shape -- shape of the images of the dataset

    classes -- integer, number of classes



    Returns:

    model -- a Model() instance in Keras

    

    Source kernel : Bengali Graphemes_ Multi Output ResNet-50. 

    The model was modified to correspond to ResNet18 Architecture.

    """



    # Define the input as a tensor with shape input_shape

    X_input = Input(input_shape)



    # Zero-Padding

    X = ZeroPadding2D((3, 3))(X_input)



    # Stage 1

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name='bn_conv1')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    

    # Stage 2

    X = convolutional_block(X, f=3, filters=64, stage=2, block='a', s=1)

    X = identity_block(X, 3, 64, stage=2, block='b')



    # Stage 3

    X = convolutional_block(X, f=3, filters=128, stage=3, block='a', s=2)

    X = identity_block(X, 3, 128, stage=3, block='b')



    # Stage 4

    X = convolutional_block(X, f=3, filters=256, stage=4, block='a', s=2)

    X = identity_block(X, 3, 256, stage=4, block='b')



    # Stage 5

    X = X = convolutional_block(X, f=3, filters=512, stage=5, block='a', s=2)

    X = identity_block(X, 3, 512, stage=5, block='b')



    # AVGPOOL

    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)



    # output layers

    X = Flatten()(X)

    head_root = Dense(168, activation = 'softmax', kernel_initializer=glorot_uniform(seed=0))(X)

    head_vowel = Dense(11, activation = 'softmax', kernel_initializer=glorot_uniform(seed=0))(X)

    head_consonant = Dense(7, activation = 'softmax', kernel_initializer=glorot_uniform(seed=0))(X)



    # Create model

    model = Model(inputs=X_input, outputs=[head_root, head_vowel, head_consonant], name='ResNet18')



    return model
model = ResNet18(input_shape=(64, 64, 1))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
IMG_SIZE=64

N_CHANNELS=1

batch_size = 512

epochs = 23
# Set a learning rate annealer. Learning rate will be half after 3 epochs if accuracy is not increased

learning_rate_reduction_root = ReduceLROnPlateau(monitor='dense_1_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)

learning_rate_reduction_vowel = ReduceLROnPlateau(monitor='dense_2_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)

learning_rate_reduction_consonant = ReduceLROnPlateau(monitor='dense_3_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)
# Stop the training if the global loss function stops decreasing (no progress in 10 epochs)

early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, 

                                                              restore_best_weights=True, mode="min")  
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
histories = []

for i in range(4):

    train_df = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)

    

    # Visualize few samples of current training dataset

    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 8))

    count=0

    for row in ax:

        for col in row:

            col.imshow(resize(train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).iloc[[count]], need_progress_bar=False).values.reshape(64, 64))

            count += 1

    plt.show()

    

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

        rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.15, # Randomly zoom image 

        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





    # This will just calculate parameters required to augment the given data. This won't perform any augmentations

    datagen.fit(x_train)



    # Fit the model

    history = model.fit_generator(datagen.flow(x_train, {'dense_1': y_train_root, 'dense_2': y_train_vowel, 'dense_3': y_train_consonant}, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]), 

                              steps_per_epoch=x_train.shape[0] // batch_size, 

                              callbacks=[learning_rate_reduction_root, learning_rate_reduction_vowel, learning_rate_reduction_consonant, early_stopping_cb])



    histories.append(history)

    

    # Delete to reduce memory usage

    del x_train

    del x_test

    del y_train_root

    del y_test_root

    del y_train_vowel

    del y_test_vowel

    del y_train_consonant

    del y_test_consonant

    gc.collect()
%matplotlib inline

def plot_loss(his, title):

    """Function which plots the history of training, in this case the evolution of training and validation loss function.

     

    ARGS : 

    - his : keras history object

    - title : str with title of each plot

    

    OUT :

    - plot of training curve    

    """



    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0, len(his.history['loss'])), his.history['loss'], label='train_loss')

    plt.plot(np.arange(0, len(his.history['loss'])), his.history['dense_1_loss'], label='train_root_loss')

    plt.plot(np.arange(0, len(his.history['loss'])), his.history['dense_2_loss'], label='train_vowel_loss')

    plt.plot(np.arange(0, len(his.history['loss'])), his.history['dense_3_loss'], label='train_consonant_loss')

    

    plt.plot(np.arange(0, len(his.history['loss'])), his.history['val_dense_1_loss'], label='val_train_root_loss')

    plt.plot(np.arange(0, len(his.history['loss'])), his.history['val_dense_2_loss'], label='val_train_vowel_loss')

    plt.plot(np.arange(0, len(his.history['loss'])), his.history['val_dense_3_loss'], label='val_train_consonant_loss')

    

    plt.title(title)

    plt.xlabel('Epoch #/' + str(len(his.history['loss'])))

    plt.ylabel('Loss')

    plt.legend(loc='upper right')

    plt.show()



def plot_acc(his, title):

    """Function which plots the history of training, in this case the evolution of training and validation accuracy.

     

    ARGS : 

    - his : keras history object

    - title : str with title of each plot



    OUT :

    - plot of training curve    

    """

    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0, len(his.history['dense_1_accuracy'])), his.history['dense_1_accuracy'], label='train_root_acc')

    plt.plot(np.arange(0, len(his.history['dense_1_accuracy'])), his.history['dense_2_accuracy'], label='train_vowel_acc')

    plt.plot(np.arange(0, len(his.history['dense_1_accuracy'])), his.history['dense_3_accuracy'], label='train_consonant_acc')

    

    plt.plot(np.arange(0, len(his.history['dense_1_accuracy'])), his.history['val_dense_1_accuracy'], label='val_root_acc')

    plt.plot(np.arange(0, len(his.history['dense_1_accuracy'])), his.history['val_dense_2_accuracy'], label='val_vowel_acc')

    plt.plot(np.arange(0, len(his.history['dense_1_accuracy'])), his.history['val_dense_3_accuracy'], label='val_consonant_acc')

    plt.title(title)

    plt.xlabel('Epoch # /' + str(len(his.history['dense_1_accuracy'])))

    plt.ylabel('Accuracy')

    plt.legend(loc='upper right')

    plt.show()
# Plot the learning curves for 4 datasets

for dataset in range(4):

    plot_loss(histories[dataset], f'Training Dataset: {dataset}')

    plot_acc(histories[dataset], f'Training Dataset: {dataset}')
# Delete histories to clean the memory

del histories

gc.collect()
# Create dictionnary of predictions

preds_dict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}
# Generate the submission .csv file

components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target=[] # model predictions placeholder

row_id=[] # row_id place holder

for i in range(4):

    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

    df_test_img.set_index('image_id', inplace=True)



    X_test = resize(df_test_img, need_progress_bar=False)/255

    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

    

    preds = model.predict(X_test)



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