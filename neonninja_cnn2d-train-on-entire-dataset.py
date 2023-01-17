import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
df_train = pd.read_csv('../input/train.csv')
X_train = df_train.drop('label', axis = 1)

y_train = df_train['label']

X_train = X_train / 255

X_train = X_train.values.reshape(-1,28,28,1)

y_train = to_categorical(y_train)

train_datagen = ImageDataGenerator(

        rotation_range=10,

        zoom_range = 0.1, 

        width_shift_range=0.1,

        height_shift_range=0.1,

        shear_range=0.1,

        )



train_datagen.fit(X_train)

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(28, 28, 1)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Flatten(),

    

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    

    tf.keras.layers.Dense(10, activation='softmax')

])





model.compile(loss='categorical_crossentropy',

              optimizer=Adam(0.0001),

              metrics=['acc'])
batch_size = 32

history = model.fit_generator(

        train_datagen.flow(X_train, y_train, batch_size=batch_size),

        epochs=20,

        )
test = pd.read_csv('../input/test.csv')

test = test / 255

test = test.values.reshape(-1,28,28,1)

pred = model.predict(test)

pred = np.argmax(pred, axis = 1)

pred_csv = pd.DataFrame(pred, columns= ['Label'])

pred_csv.index += 1

print(pred_csv)

pred_csv.to_csv('submission.csv', index_label='ImageId' )