import csv

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
# configure based on your local data set path

kaggle_path='/kaggle/input/Kannada-MNIST'

local_path='/tmp/Kannada-MNIST'

local_path=kaggle_path

df = pd.read_csv(local_path+'/train.csv')

train, test = train_test_split(df, test_size=0.3)
def get(data):

    temp_images = []

    temp_labels = []

    for index, row in data.iterrows():

        temp_labels.append(row[0])

        image_data = row[1:]

        image_data_as_array = np.array_split(image_data, 28)

        temp_images.append(image_data_as_array)

    images = np.array(temp_images).astype('float')

    labels = np.array(temp_labels).astype('float')

    return images, labels

training_images, training_label = get(train)

test_images, test_label = get(test)
training_images = np.expand_dims(training_images, axis=3)

testing_images = np.expand_dims(test_images, axis=3)



train_datagen = ImageDataGenerator(

    rescale=1. / 255)



validation_datagen = ImageDataGenerator(

    rescale=1. / 255)



print(training_images.shape)

print(testing_images.shape)
training_images.shape
class myCallback(tf.keras.callbacks.Callback):

      def on_epoch_end(self, epoch, logs={}):

        if(logs.get('accuracy')>0.96):

            print("\nReached 99.99% accuracy so cancelling training!")

            self.model.stop_training = True



callbacks = myCallback()        

        

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])



model.compile(optimizer = tf.optimizers.Adam(),

              loss = 'sparse_categorical_crossentropy',

              metrics=['accuracy'])



history = model.fit_generator(train_datagen.flow(training_images, training_label, batch_size=32),

                              steps_per_epoch=len(training_images) / 32,

                              epochs=100,

                              validation_data=validation_datagen.flow(testing_images, test_label, batch_size=100),

                              validation_steps=len(testing_images) / 32,

                             callbacks=[callbacks])



model.evaluate(testing_images, test_label)

# Reading the test File to genreate predictions

test_data = pd.read_csv(local_path+"/test.csv", sep=",")

test_data.pop("id")

x_test = np.array(test_data.values).astype('float')

print(x_test.shape)
# Preparint the test input for the predictions

x_test = [i.reshape(28,28, 1) for i in x_test]

predictions = model.predict([x_test], batch_size=5000)
# save and send to Kaggle 

def to_csv(predictions):

    with open("submission.csv", "w") as out:

        out.write("id,label\n")

        for i in range(len(predictions)):

            out.write(str(i)+","+str(np.unravel_index(np.argmax(predictions[i], axis=None), predictions[i].shape)[0])+"\n")

            

    return True



to_csv(predictions)