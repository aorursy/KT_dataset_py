# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import gc



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from tensorflow import keras

import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.models import clone_model

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont

from matplotlib import pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

data0 = pd.read_feather('/kaggle/input/bengali-preprocessed-dataset/train_data_0.feather')

data1 = pd.read_feather('/kaggle/input/bengali-preprocessed-dataset/train_data_1.feather')

data2 = pd.read_feather('/kaggle/input/bengali-preprocessed-dataset/train_data_2.feather')

data3 = pd.read_feather('/kaggle/input/bengali-preprocessed-dataset/train_data_3.feather')
data_full = pd.concat([data0,data1,data2,data3],ignore_index=True)
train_df = pd.merge(data_full, train, on='image_id').drop(['image_id'], axis=1).drop(['grapheme'], axis=1)
train_df
IMG_SIZE=64

N_CHANNELS=1
def get_dummies(df):

    cols = []

    for col in df:

        cols.append(pd.get_dummies(df[col].astype(str)))

    return pd.concat(cols, axis=1)
inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 1))



model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1))(inputs)

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.9)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = Dropout(rate=0.3)(model)



model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.9)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.9)(model)

model = Dropout(rate=0.3)(model)



model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.9)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.9)(model)

model = Dropout(rate=0.3)(model)



model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.9)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=256, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.9)(model)

model = Dropout(rate=0.3)(model)



model = Flatten()(model)

model = Dense(1024, activation = "relu")(model)

model = Dropout(rate=0.3)(model)

dense = Dense(512, activation = "relu")(model)



head_root = Dense(168, activation = 'softmax')(dense)

head_vowel = Dense(11, activation = 'softmax')(dense)

head_consonant = Dense(7, activation = 'softmax')(dense)



model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Set a learning rate annealer. Learning rate will be half after 3 epochs if accuracy is not increased

learning_rate_reduction_root = ReduceLROnPlateau(monitor='dense_3_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)

learning_rate_reduction_vowel = ReduceLROnPlateau(monitor='dense_4_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)

learning_rate_reduction_consonant = ReduceLROnPlateau(monitor='dense_5_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)
batch_size = 800

epochs = 65
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
HEIGHT = 137

WIDTH = 236
X_train = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)

X_train = X_train.astype(np.float16)/255

gc.collect()



# CNN takes images in shape `(batch_size, h, w, channels)`, so reshape the images

X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)



Y_train_root = pd.get_dummies(train_df['grapheme_root']).values

Y_train_vowel = pd.get_dummies(train_df['vowel_diacritic']).values

Y_train_consonant = pd.get_dummies(train_df['consonant_diacritic']).values



print(f'Training images: {X_train.shape}')

print(f'Training labels root: {Y_train_root.shape}')

print(f'Training labels vowel: {Y_train_vowel.shape}')

print(f'Training labels consonants: {Y_train_consonant.shape}')



del train_df

gc.collect()



# Data augmentation for creating more training data

datagen = MultiOutputDataGenerator(

    featurewise_center=False,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization=False,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    zca_whitening=False,  # apply ZCA whitening

    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

    zoom_range = 0.2, # Randomly zoom image 

    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

    horizontal_flip=False,  # randomly flip images

    vertical_flip=False)  # randomly flip images





# This will just calculate parameters required to augment the given data. This won't perform any augmentations

datagen.fit(X_train)



# Fit the model

history = model.fit_generator(datagen.flow(X_train, {'dense_3': Y_train_root, 'dense_4': Y_train_vowel, 'dense_5': Y_train_consonant}, batch_size=batch_size),

                          epochs = epochs,

                          steps_per_epoch=X_train.shape[0] // batch_size, 

                          callbacks=[learning_rate_reduction_root, learning_rate_reduction_vowel, learning_rate_reduction_consonant])



gc.collect()
%matplotlib inline

def plot_loss(his, epoch, title):

    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')

    plt.plot(np.arange(0, epoch), his.history['dense_3_loss'], label='train_root_loss')

    plt.plot(np.arange(0, epoch), his.history['dense_4_loss'], label='train_vowel_loss')

    plt.plot(np.arange(0, epoch), his.history['dense_5_loss'], label='train_consonant_loss')

    

    plt.title(title)

    plt.xlabel('Epoch #')

    plt.ylabel('Loss')

    plt.legend(loc='upper right')

    plt.show()



def plot_acc(his, epoch, title):

    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history['dense_3_accuracy'], label='train_root_acc')

    plt.plot(np.arange(0, epoch), his.history['dense_4_accuracy'], label='train_vowel_accuracy')

    plt.plot(np.arange(0, epoch), his.history['dense_5_accuracy'], label='train_consonant_accuracy')



    plt.title(title)

    plt.xlabel('Epoch #')

    plt.ylabel('Accuracy')

    plt.legend(loc='upper right')

    plt.show()
plot_loss(history, epochs, f'Training Loss')

plot_acc(history, epochs, f'Training Accuracy')
model.save('model_65epoch_batch_normalized1.h5')