import tensorflow as tf

import tensorflow.keras as k

import numpy as np

from tensorflow.keras.datasets.mnist import load_data

from tensorflow.keras.preprocessing.image import ImageDataGenerator
num_classes = 10

(x_train, y_train), (x_test, y_test) = load_data()



x_train = np.expand_dims(x_train, -1)

y_train = k.utils.to_categorical(y_train, num_classes)



x_test = np.expand_dims(x_test, -1)

y_test = k.utils.to_categorical(y_test, num_classes)
datagen = ImageDataGenerator(featurewise_center=False,

                            featurewise_std_normalization=False,

                            rotation_range=20,

                            width_shift_range=0.2,

                            height_shift_range=0.2)
batch_size = 128

train_generator = datagen.flow( x_train,

                                y=y_train,

                                batch_size=batch_size,

                                shuffle=True)

valid_generator = datagen.flow( x_test,

                                y=y_test,

                                batch_size=batch_size,

                                shuffle=False)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import add, Dense, Conv2D, Flatten, MaxPool2D, Dropout
model = Sequential()

model.add(Conv2D(6, (5, 5), activation='sigmoid', input_shape = (28,28,1)))

model.add(MaxPool2D((2, 2), strides=(2,2)))

model.add(Conv2D(16, (5,5), activation='sigmoid'))

model.add(MaxPool2D((2, 2), strides=(2,2)))

model.add(Conv2D(120, (4, 4), activation='sigmoid'))

model.add(Flatten())

model.add(Dense(84,activation='sigmoid'))

model.add(Dense(num_classes, activation='softmax'))



model.summary()



model.compile(optimizer=k.optimizers.RMSprop(lr=2e-3, decay=1e-5),

                loss='categorical_crossentropy',

                metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 

                                            min_delta=0.01,

                                            verbose=1,

                                            patience=2,

                                            mode="auto",

                                            baseline=0.96)



fit_stats = model.fit_generator(train_generator,

                                steps_per_epoch=60000//batch_size,

                                validation_data=valid_generator,

                                validation_steps=10000//batch_size,

                                callbacks=[callback],

                                epochs=50)