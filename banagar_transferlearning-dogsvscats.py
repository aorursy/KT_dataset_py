import os

import keras

import numpy as np

import pandas as pd

from keras import models

from keras import optimizers

from keras.models import Sequential

from keras.applications import VGG16

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
filenames = os.listdir("../input/dogs-vs-cats/train/train")

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append('dog')

    else:

        categories.append('cat')



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
conv_base = VGG16(weights='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',

                  include_top=False,

                  input_shape=(150, 150, 3))

conv_base.summary()
model = models.Sequential()

model.add(conv_base)

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()
print('This is the number of trainable weights '

      'before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False

print('This is the number of trainable weights '

      'after freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:

    if layer.name == 'block5_conv1':

        set_trainable = True

    if set_trainable:

        layer.trainable = True

    else:

        layer.trainable = False
train_datagen = ImageDataGenerator(

      rescale=1./255,

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')



# Note that the validation data should not be augmented!

validation_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_dataframe(

        # This is the target directory

        train_df, 

        "../input/dogs-vs-cats/train/train", 

        x_col='filename',

        y_col='category',

        # All images will be resized to 150x150

        target_size=(150, 150),

        batch_size=200,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')



validation_generator = validation_datagen.flow_from_dataframe(

        validate_df, 

        "../input/dogs-vs-cats/train/train", 

        x_col='filename',

        y_col='category',

        target_size=(150, 150),

        batch_size=100,

        class_mode='binary')
model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-5),

              metrics=['acc'])



history = model.fit_generator(

      train_generator,

      steps_per_epoch=100,

      epochs=30,

      validation_data=validation_generator,

      validation_steps=50)
import matplotlib.pyplot as plt

%matplotlib inline



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
model.save('dogvscat.h5')