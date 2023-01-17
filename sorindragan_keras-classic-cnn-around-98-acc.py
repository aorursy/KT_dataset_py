import os

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import ImageDataGenerator



data_path = '/kaggle/input/rockpaperscissors/rps-cv-images/'



print(os.listdir(data_path))



train_paper_dir = os.path.join(data_path + 'paper')

train_rock_dir = os.path.join(data_path + 'rock')

train_scissors_dir = os.path.join(data_path + 'scissors')



# visualizing the data

n = 3

for f in [train_rock_dir, train_paper_dir, train_scissors_dir]:

    for i in range(n):

        sp = plt.subplot(1, n, i + 1)

        sp.axis('Off')

        img = mpimg.imread(f + "/" + os.listdir(f)[i])

        plt.imshow(img)

    plt.show()
# callbacks class definition

class callBack(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if (logs.get('loss') < 0.04):

            print()

            print("Reached almost 99% accuracy so cancelling training!")

            self.model.stop_training = True





callbacks = callBack()
# model creation

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 300, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),



    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),



    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),



    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),



    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1024, activation='relu'),



    tf.keras.layers.Dense(3, activation='softmax')

])



model.summary()



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy']

              )
# train/validation split using ImageDataGenerator

_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)



train_generator = _datagen.flow_from_directory(

    data_path,

    target_size=(200, 300),

    batch_size=128,

    class_mode='categorical',

    subset='training'

)



validation_generator = _datagen.flow_from_directory(

    data_path,

    target_size=(200, 300),

    batch_size=32,

    class_mode='categorical',

    subset='validation'

)
# the first 3 images with the corresponding labels

x, y = train_generator.next()

plt.imshow(x[0], interpolation='nearest')

plt.show()

print(y[0])



plt.imshow(x[1], interpolation='nearest')

plt.show()

print(y[1])



plt.imshow(x[2], interpolation='nearest')

plt.show()

print(y[2])
# model training

history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_generator.samples / train_generator.batch_size - 1,

    epochs=15,

    verbose=1,

    validation_data=validation_generator,

    validation_steps=validation_generator.samples / validation_generator.batch_size - 1,

    callbacks=[callbacks]

)



print(history.epoch, history.history['acc'][-1])
# Trainig Loss/Acc and Validation Loss/Acc visualization

plt.figure(figsize=(20,10))

plt.subplot(1, 2, 1)

plt.ylabel('Loss', fontsize=16)

plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.legend(loc='upper right')



plt.subplot(1, 2, 2)

plt.ylabel('Accuracy', fontsize=16)

plt.plot(history.history['acc'], label='Training Accuracy')

plt.plot(history.history['val_acc'], label='Validation Accuracy')

plt.legend(loc='lower right')

plt.show()