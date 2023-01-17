from keras.models import Sequential                             # To specify a sequential model. 

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense   # Components of the CNN model, we will cover them in details later.

from keras.preprocessing.image import ImageDataGenerator        # To load image data and augment it with more examples.



# General purpose tools.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
classifier = Sequential()
classifier.add(

    Conv2D( 32, (3,3),

           input_shape = (64, 64, 3),

           activation = 'relu')

)
classifier.add(

    MaxPooling2D(

        pool_size = (2,2)

    )

)
classifier.add(Flatten())
classifier.add(

    Dense(

        units = 128, 

        activation = 'relu'

    )

)
classifier.add(

    Dense(

        units = 1, 

        activation = 'sigmoid'

    )

)
classifier.compile(

    optimizer = 'adam',

    loss = 'binary_crossentropy',

    metrics = ['accuracy']

)
classifier.summary()
train_datagen = ImageDataGenerator(

    rescale = 1./255,

    shear_range = 0.2,

    zoom_range = 0.2,

    horizontal_flip = True

)



test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(

    '/kaggle/input/cat-and-dog/training_set/training_set',

    target_size = (64,64),

    batch_size = 32,

    class_mode = 'binary'

)
test_set = test_datagen.flow_from_directory(

    '/kaggle/input/cat-and-dog/test_set/test_set',

    target_size = (64,64),

    batch_size = 32,

    class_mode = 'binary'

)
history = classifier.fit_generator(

    training_set,

    steps_per_epoch = 2500,

    epochs = 5,

    validation_data = test_set,

    validation_steps = 1000

)
classifier.save('cats_and_dogs_classifier_1.h5')
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and Validation Accuracy')

plt.legend()



plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and Validation Loss')

plt.legend()



plt.show()