# Import Tensorflow 2.0

import tensorflow as tf 



import matplotlib.pyplot as plt

import numpy as np

import random

from tqdm import tqdm



# Check that we are using a GPU, if not switch runtimes

#   using Runtime > Change Runtime Type > GPU

assert len(tf.config.list_physical_devices('GPU')) > 0
import os

os.listdir('../input/digit-recognizer')
import pandas as pd



base = '../input/digit-recognizer/'

train_df = pd.read_csv(base+'train.csv')

test_df = pd.read_csv(base+'test.csv')



# df.head(1)
df = pd.concat([train_df, test_df])

cols = list(df.columns)

cols.remove('label')

df[cols] = df[cols] / 255.0

df.shape
train = df.iloc[:len(train_df)]

test = df.iloc[len(train_df):]

test.drop(columns=['label'], inplace=True)



X_train = train.drop(columns=['label'])

Y_train = train['label']

X_train = X_train.values.reshape(-1,28,28,1).astype(np.float32)

X_test = test.values.reshape(-1,28,28,1).astype(np.float32)
from sklearn.model_selection import train_test_split



X_sample, X_val, Y_sample, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state = 7)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_sample)
def build_cnn_model():

    cnn_model = tf.keras.Sequential([

        

#         tf.keras.layers.Flatten(),

        tf.keras.layers.Conv2D(filters=36, kernel_size=(5,5), activation=tf.nn.relu),

#         tf.keras.layers.Conv2D(filters=36, kernel_size=(3,3), activation=tf.nn.relu),

        tf.keras.layers.MaxPool2D(),

#         tf.keras.layers.Dropout(0.1),

        

        tf.keras.layers.Conv2D(filters=36, kernel_size=(5,5), activation=tf.nn.relu),

#         tf.keras.layers.Conv2D(filters=36, kernel_size=(5,5), activation=tf.nn.relu),

        tf.keras.layers.MaxPool2D(),

#         tf.keras.layers.Dropout(0.1),

        

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation=tf.nn.relu),

        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(10, activation=tf.nn.softmax)

        

    ])

    

    return cnn_model



cnn_model = build_cnn_model()

cnn_model.predict(X_sample[:3])

print(cnn_model.summary())
cnn_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.05), loss='sparse_categorical_crossentropy', metrics=['accuracy'])



BATCH_SIZE = 40



# cnn_model.fit(X_sample, Y_sample, batch_size = BATCH_SIZE, epochs=20)



history = cnn_model.fit_generator(datagen.flow(X_sample,Y_sample, batch_size=BATCH_SIZE),

                              epochs = 50, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_sample.shape[0] // BATCH_SIZE)
test_loss, test_acc = cnn_model.evaluate(X_val, Y_val, batch_size=BATCH_SIZE)

print("Test Accuracy: {:.4f}".format(test_acc))
predictions = cnn_model.predict(X_test)

predictions[1]
predictions = [np.argmax(predictions[i]) for i in range(len(predictions))]

predictions[:3]
sample = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

sample.head(3)
sample['Label'] = predictions

sample.head(2)
sample.to_csv('CNN_2layers_Adagrad4.csv', index=False)