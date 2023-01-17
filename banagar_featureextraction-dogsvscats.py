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
datagen = ImageDataGenerator(rescale=1./255)

batch_size = 20



def extract_features(directory, sample_count):

    features = np.zeros(shape=(sample_count, 4, 4, 512))

    labels = np.zeros(shape=(sample_count))

    generator = datagen.flow_from_dataframe(

        directory,

        "../input/dogs-vs-cats/train/train",

        x_col='filename',

        y_col='category',

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')

    i = 0

    for inputs_batch, labels_batch in generator:

        features_batch = conv_base.predict(inputs_batch)

        features[i * batch_size : (i + 1) * batch_size] = features_batch

        labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        i += 1

        if i * batch_size >= sample_count:

            # Note that since generators yield data indefinitely in a loop,

            # we must `break` after every image has been seen once.

            break

    return features, labels



train_features, train_labels = extract_features(train_df, 20000)

validation_features, validation_labels = extract_features(validate_df, 5000)
train_features = np.reshape(train_features, (20000, 4 * 4 * 512))

validation_features = np.reshape(validation_features, (5000, 4 * 4 * 512))
model = models.Sequential()

model.add(Dense(256, activation='relu', input_dim=4 * 4 * 512))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer=optimizers.RMSprop(lr=2e-5),

              loss='binary_crossentropy',

              metrics=['acc'])



history = model.fit(train_features, train_labels,

                    epochs=30,

                    batch_size=20,

                    validation_data=(validation_features, validation_labels))
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