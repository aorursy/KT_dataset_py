import os

import warnings



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



from keras.preprocessing.image import ImageDataGenerator
BASE_DIR = '../input/'

TRAIN_DIR = os.path.join(BASE_DIR, 'train/train')

TEST_DIR = os.path.join(BASE_DIR, 'test1/test1')



os.listdir(BASE_DIR)
def make_dataframe_from_dir(path):

    filenames = os.listdir(path)

    categories = []

    

    for filename in filenames:

        if filename.split('.')[0] == 'dog':

            categories.append(1)

        else:

            categories.append(0)



    df = pd.DataFrame(

        {

            'filename': filenames,

            'category': categories

        }

    )

    

    return df

    

    

df = make_dataframe_from_dir(TRAIN_DIR)



print(df.shape)

df.tail()
train_df, validation_df = train_test_split(df, test_size=0.20,random_state=43)
print(train_df.shape)

print(validation_df.shape)
from keras import layers

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
from keras import optimizers



model.compile(loss='binary_crossentropy',

             optimizer=optimizers.RMSprop(lr=1e-4),

             metrics=['acc'])
train_df.category = train_df.category.astype('str')

validation_df.category = validation_df.category.astype('str')
total_train = train_df.shape[0]

total_validation = validation_df.shape[0]

BATCH_SIZE = 20



print("total train {0}, validation {1}".format(total_train, total_validation))

print("batch size: ", BATCH_SIZE)
train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_dataframe(

    train_df,

    TRAIN_DIR,

    x_col='filename',

    y_col='category',

    target_size=(150, 150),

    batch_size=BATCH_SIZE,

    class_mode='binary'

)



validation_generator = test_datagen.flow_from_dataframe(

    validation_df,

    TRAIN_DIR,

    x_col='filename',

    y_col='category',

    target_size=(150, 150),

    batch_size=BATCH_SIZE,

    class_mode='binary'

)
history = model.fit_generator(

    train_generator,

    steps_per_epoch=total_train//BATCH_SIZE,

    epochs=20,

    validation_data=validation_generator,

    validation_steps=total_validation//BATCH_SIZE

)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Training acc')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'bo', label='Traiing loss')

plt.plot(epochs, val_loss, 'b', label='Traiing loss')

plt.legend()



plt.show()
test_df = make_dataframe_from_dir(TEST_DIR)



test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df,

    TEST_DIR,

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=(150, 150),

    batch_size=BATCH_SIZE,

    shuffle=False

)