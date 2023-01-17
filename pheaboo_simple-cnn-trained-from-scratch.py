import pandas as pd

import numpy as np

import pylab as plt

%matplotlib inline



import tensorflow as tf

from tensorflow import keras



from tqdm import tqdm

import random

random.seed(0)



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import InputLayer, Input

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Activation

from tensorflow.keras.layers import BatchNormalization, Reshape, MaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
batch_size = 64

target_size = (32, 32)

class_mode = 'binary'

epochs = 100

input_shape = (32,32,3)

num_classes = 2

data_dir =  "../input/train/train/"

validation_split = 0.8

color_mode='rgb'

x_col = 'id'

y_col='has_cactus'

dropout_dense_layer = 0.5
df = pd.read_csv('../input/train.csv')

df.has_cactus = df.has_cactus.astype(str) # Classes must be str and not int

msk = np.random.rand(len(df)) < validation_split

train = df[msk]

validation = df[~msk]
train_datagen = ImageDataGenerator(rescale=1/255, horizontal_flip=True, vertical_flip=True)

train_generator = train_datagen.flow_from_dataframe(train, directory=data_dir, x_col=x_col, y_col=y_col, target_size=target_size, color_mode=color_mode, class_mode=class_mode, batch_size=batch_size, shuffle=True)

validation_generator = train_datagen.flow_from_dataframe(validation, directory=data_dir, x_col=x_col, y_col=y_col, target_size=target_size, color_mode=color_mode, class_mode=class_mode, batch_size=batch_size, shuffle=True)
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)

class_weights
model = Sequential()



model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))

model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))

model.add(BatchNormalization())

model.add(MaxPooling2D())



model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D())



model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D())



model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D())



model.add(GlobalAveragePooling2D())



model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(dropout_dense_layer))



model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss=keras.losses.binary_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])



model.summary()



callbacks = [EarlyStopping(monitor='val_loss', patience=20),

             ReduceLROnPlateau(patience=10, verbose=1),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', verbose=0, save_best_only=True)]



history = model.fit_generator(train_generator,

          validation_data=validation_generator,

          epochs=epochs,

          verbose=1,

          shuffle=True,

          callbacks=callbacks,

          class_weight=class_weights)
plt.figure(figsize=(15,5))



plt.subplot(141)

plt.plot(history.history['loss'], label='training')

plt.plot(history.history['val_loss'], label='validation')

plt.xlabel('# Epochs')

plt.legend()

plt.ylabel("Loss - Binary Cross Entropy")

plt.title('Loss Evolution')



plt.subplot(142)

plt.plot(history.history['loss'], label='training')

plt.plot(history.history['val_loss'], label='validation')

plt.ylim(0,0.1)

plt.xlabel('# Epochs')

plt.legend()

plt.ylabel("Loss - Binary Cross Entropy")

plt.title('Zoom Near Zero - Loss Evolution')



plt.subplot(143)

plt.plot(history.history['acc'], label='training')

plt.plot(history.history['val_acc'], label='validation')

plt.xlabel('# Epochs')

plt.ylabel("Accuracy")

plt.legend()

plt.title('Accuracy Evolution')



plt.subplot(144)

plt.plot(history.history['acc'], label='training')

plt.plot(history.history['val_acc'], label='validation')

plt.ylim(0.98,1)

plt.xlabel('# Epochs')

plt.ylabel("Accuracy")

plt.legend()

plt.title('Zoom Near One - Accuracy Evolution')
model.load_weights("best_model.h5")



history.history['val_acc'][np.argmin(history.history['val_loss'])]
test_folder = "../input/test/"

test_datagen = ImageDataGenerator(

    rescale=1. / 255)



test_generator = test_datagen.flow_from_directory(

    directory=test_folder,

    target_size=target_size,

    batch_size=1,

    class_mode=None,

    shuffle=False)
sample_submission = pd.read_csv('../input/sample_submission.csv')

filenames = [path.split('/')[-1] for path in test_generator.filenames]

probabilities = list(model.predict_generator(test_generator)[:,0])



sample_submission.id = filenames

sample_submission.has_cactus = probabilities



sample_submission.to_csv('sample_submission.csv', index=False)