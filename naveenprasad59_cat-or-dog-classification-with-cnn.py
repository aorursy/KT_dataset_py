import tensorflow as tf 

import tensorflow.keras # tensorflow and keras are deep learning API's which makes our Deep Learning process Simple

from tensorflow.keras import Sequential  #this package is for stacking the layers of my model

from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout #these are the differnet types of layers in the Deep learning model

from tensorflow.keras.preprocessing.image import ImageDataGenerator #this a package used to scale our dataset images

from tensorflow.keras.optimizers import RMSprop #this is an type of optimizer and this is a algorithm to minimize loss

import matplotlib.pyplot as plt #this package is used to plot our data or result
TRAINING_DIR = '../input/cat-and-dog/training_set/training_set/'

train_datagen = ImageDataGenerator(

                    rescale=1./255,

                    rotation_range=40,

                    width_shift_range=0.2,

                    height_shift_range=0.2,

                    shear_range=0.2,

                    zoom_range=0.2,

                    horizontal_flip=True,

                    fill_mode='nearest')

train_generator =train_datagen.flow_from_directory(

                   TRAINING_DIR,

                   target_size=(224,224),

                   batch_size=64,

                   class_mode='binary'

                     )
VALIDATION_DIR = '../input/cat-and-dog/test_set/test_set/'

validation_datagen = ImageDataGenerator(

                           rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(

                            VALIDATION_DIR,

                            target_size=(224,224),

                            batch_size=32,

                            class_mode='binary')
model = tf.keras.models.Sequential([

    Conv2D(16,(3,3),activation='relu',input_shape=(224,224,3)),

    MaxPool2D(2,2),

    Conv2D(32,(3,3),activation='relu'),

    Conv2D(32,(3,3),activation='relu'),

    MaxPool2D(2,2),

    Conv2D(64,(3,3),activation='relu'),

    MaxPool2D(2,2),

    Dropout(0.3),

    Flatten(),

    Dense(256,activation='relu'),

    Dense(128,activation='relu'),

    Dense(64,activation='relu'),

    Dropout(0.25),

    Dense(1,activation='sigmoid')

])



model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
history = model.fit(train_generator,

                              epochs=30,

                              steps_per_epoch=125,

                              verbose=1,

                              validation_data=validation_generator)
acc=history.history['accuracy']

val_acc=history.history['val_accuracy']

loss=history.history['loss']

val_loss=history.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()