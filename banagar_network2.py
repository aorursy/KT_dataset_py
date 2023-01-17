import os

import numpy as np

import pandas as pd

filenames = os.listdir('../input/beheshticrops')

df = pd.DataFrame({

    'filename': filenames,

})

df['exist'] = np.nan

df2 = pd.read_excel('../input/beheshticrops/BeheshtiCrops.xlsx')

for i in range(404):

    for j in range(400):

        if df['filename'][i]==df2['filename'][j]:

            df['exist'][i]=1

df = df.drop(df.index[df['exist'].isnull()])

df = df.drop('exist', axis=1)

df = df.merge(df2, on='filename', how='left')

print(df.shape)

df.head()
import keras

from keras import models

from keras import optimizers

from keras.optimizers import Adam

from keras.models import Sequential

from keras.applications import VGG16

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
train_df, validate_df = train_test_split(df, test_size=0.25, random_state=8)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
conv_base = VGG16(weights='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',

                  include_top=False,

                  input_shape=(150, 150, 3))

conv_base.summary()
model = models.Sequential()

model.add(conv_base)

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(1, activation='linear'))

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

      rotation_range=15,

      width_shift_range=0.01,

      horizontal_flip=True,

      fill_mode='nearest')



# Note that the validation data should not be augmented!

validation_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_dataframe(

        # This is the target directory

        train_df, 

        "../input/beheshticrops", 

        x_col='filename',

        y_col='label',

        # All images will be resized to 150x150

        target_size=(150, 150),

        batch_size=5,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='raw')



validation_generator = validation_datagen.flow_from_dataframe(

        validate_df, 

        "../input/beheshticrops", 

        x_col='filename',

        y_col='label',

        target_size=(150, 150),

        batch_size=5,

        class_mode='raw')
opt = Adam(lr=1e-4, decay=1e-1 / 125)

model.compile(loss="mean_absolute_percentage_error", optimizer=opt, metrics=['mean_absolute_percentage_error'])
history = model.fit_generator(

      train_generator,

      steps_per_epoch=60,

      epochs=125,

      validation_data=validation_generator,

      validation_steps=20)
import matplotlib.pyplot as plt

%matplotlib inline



acc = history.history['mean_absolute_percentage_error']

val_acc = history.history['val_mean_absolute_percentage_error']

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
model.save('DropRegression.h5')