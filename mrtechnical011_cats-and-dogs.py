from tensorflow.keras.models import Sequential

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('../input/dogs-cats-images/dog vs cat/dataset/training_set',

                                                 target_size = (64, 64),

                                                 batch_size = 60,

                                                 class_mode = 'binary')



test_set = test_datagen.flow_from_directory('../input/dogs-cats-images/dog vs cat/dataset/test_set',

                                            target_size = (64, 64),

                                            batch_size = 60,

                                            class_mode = 'binary')
model=Sequential()

model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", input_shape=training_set.image_shape))

model.add(MaxPool2D(pool_size=2, strides=2, padding='same'))

model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=2, strides=2, padding='same'))

model.add(Dropout(0.2))





model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu", input_shape=training_set.image_shape))

model.add(MaxPool2D(pool_size=2, strides=2, padding='same'))

model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=2, strides=2, padding='same'))

model.add(Dropout(0.2))





model.add(Flatten())

model.add(Dense(units=500, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(units=100, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(units = len(set(training_set.classes)), activation = 'softmax'))#you can also set unit as 1 with sigmoid

model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

model.summary()
fm = model.fit_generator(training_set,epochs =50,

                        validation_data = test_set,

                       )#steps_per_epoch=total img(8000)/batch=200

model.save("model.h5")

accuracy = fm.history['accuracy']

plt.plot(range(len(accuracy)), accuracy, 'red', label = 'accuracy')

plt.legend()
from tensorflow.keras.models import load_model

model = load_model('model.h5')

model.summary()
"""test_image = image.load_img(test_img, target_size = (64, 64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = model.predict(x = test_image)

print(result)

if result[0][0]  == 1:

    prediction = 'Dog'

else:

    prediction = 'Cat'

"""