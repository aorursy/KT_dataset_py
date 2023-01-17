import pandas as pd
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

import seaborn as sns

from tqdm import tqdm
import os
# Initialize global variables
SAMPLE_SIZE = 10000
BATCH_SIZE = 32
TEST_PERC = 0.2
segmentations = pd.read_csv("../input/train_ship_segmentations.csv")
segmentations['path'] = '../input/train/' + segmentations['ImageId']
segmentations.shape
segmentations = segmentations.sample(n=SAMPLE_SIZE)
def has_ship(encoded_pixels):
    hs = [0 if pd.isna(n) else 1 for n in tqdm(encoded_pixels)]
    return hs
segmentations['HasShip'] = has_ship(segmentations['EncodedPixels'].values)
segmentations['HasShip'].head()
segmentations.head()
sns.countplot(segmentations['HasShip'])
np.shape(load_img(segmentations['path'].values[0]))
train,test = train_test_split(segmentations, test_size=TEST_PERC)
idg_train = ImageDataGenerator(rescale=1. / 255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True)

idg_test = ImageDataGenerator(rescale=1. / 255)
def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen
train_images = flow_from_dataframe(idg_train, train, 'path', 'HasShip', batch_size=BATCH_SIZE, target_size=(256, 256))
test_images = flow_from_dataframe(idg_train, test, 'path', 'HasShip', batch_size=BATCH_SIZE, target_size=(256, 256))
train_images.target_size
model = Sequential()
model.add(Convolution2D(32, (3, 3),
                       input_shape=(256, 256, 3),
                       activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3),
                       input_shape=(256, 256, 3),
                       activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3),
                       input_shape=(256, 256, 3),
                       activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=128, activation='relu', kernel_initializer='normal'))
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='normal'))

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])
model.summary()
fitted_model = model.fit_generator(train_images,
                   steps_per_epoch=SAMPLE_SIZE*(1-TEST_PERC)/BATCH_SIZE,
                   epochs=20,
                   validation_data=test_images,
                   validation_steps=SAMPLE_SIZE*(TEST_PERC)/BATCH_SIZE)
import matplotlib.pyplot as plt
import pylab


path = 'results'
name = 'adam'

plt.plot(fitted_model.history['acc'])
plt.plot(fitted_model.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.figure()
plt.gcf().clear()
plt.plot(fitted_model.history['loss'])
plt.plot(fitted_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()

