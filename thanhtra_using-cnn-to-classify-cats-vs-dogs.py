import os

import zipfile

import numpy as np

import pandas as pd

import random

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
filenames = os.listdir("../input/dogs-vs-cats/train/train")

labels = []

for name in filenames:

    label = name.split('.')[0]

    if label == 'dog':

        labels.append(1)

    else:

        labels.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'label': labels

})



df.head()
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

from keras.optimizers import RMSprop



model = Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid')) 



model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])



model.summary()

df['label'] = df['label'].replace({0:'cat', 1:'dog'})



train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
train_df.shape
validate_df.shape
train_datagen = ImageDataGenerator(rescale=1/255,

                                  rotation_range=40,

                                  width_shift_range=0.2,

                                  height_shift_range=0.2,

                                  shear_range=0.2,

                                  zoom_range=0.2,

                                  horizontal_flip=True,

                                  fill_mode='nearest')



train_generator = train_datagen.flow_from_dataframe(train_df,

                                                    '../input/dogs-vs-cats/train/train',

                                                    x_col = 'filename',

                                                    y_col = 'label',

                                                    batch_size=20,

                                                    class_mode='binary',

                                                    target_size=(150, 150))
valid_datagen = ImageDataGenerator(rescale=1/255)



valid_generator = valid_datagen.flow_from_dataframe(validate_df,

                                                   '../input/dogs-vs-cats/train/train',

                                                   x_col = 'filename',

                                                   y_col = 'label',

                                                   batch_size = 20,

                                                   class_mode = 'binary',

                                                   target_size = (150,150))
history = model.fit_generator(train_generator,

                              validation_data=valid_generator,

                              steps_per_epoch=100,

                              epochs=100,

                              validation_steps=50,

                              verbose=2)
import matplotlib

from matplotlib import pyplot as plt



# Retrieve a list of list results on training and test data sets for each training epoch

acc      = history.history[     'acc' ]

val_acc  = history.history[ 'val_acc' ]

loss     = history.history[    'loss' ]

val_loss = history.history['val_loss' ]



epochs   = range(len(acc)) # Get number of epochs



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')



plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()





test_filenames = os.listdir("../input/dogs-vs-cats/test1/test1")

test_df = pd.DataFrame({

    'id': test_filenames

})

test_df.shape
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "../input/dogs-vs-cats/test1/test1/", 

    x_col='id',

    y_col=None,

    class_mode=None,

    target_size=(150,150),

    batch_size=20

)
predict = model.predict_generator(test_generator, steps=625) # test shape/ batch_size
test_df['label'] = (predict > 0.5).astype(int)
test_df['id'] = test_df['id'].str.split('.').str[0]
test_df.to_csv('submission.csv', index=False)