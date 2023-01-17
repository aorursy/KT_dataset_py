import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import Image

import random

import logging as logger

import os

import csv



logger.basicConfig(format='%(process)d-%(levelname)s-%(message)s', level=logger.CRITICAL)





tf.enable_eager_execution()
# Dir vars

BASE_DIR = '../input/'

TRAIN_DIR = 'train/train/'

TRAIN_FILE = 'train.csv'

TEST_DIR = 'test/test/'

TEST_FILE = 'sample_submission.csv'



# Global vars

NUM_PARALLEL_CALLS = 4

BATCH_SIZE = 200
# Load csv Data

train_df = pd.read_csv(BASE_DIR + TRAIN_FILE)

train_df.head(5)
# unique values, indices

labels = train_df['Id']

unique_labels = list(set(labels))

total_unique_labels = len(unique_labels)



len(unique_labels)
len(labels)
# Read Image

Image(filename=BASE_DIR + TRAIN_DIR + random.choice(train_df['Image']))
# Create Data Pipeline

def prepare_image(imageFileName, label):

    imageFileName = imageFileName.numpy()

    logger.debug('reading image {} with label {}'.format(imageFileName, label.numpy()))

    img = tf.keras.preprocessing.image.load_img(imageFileName, target_size=(100, 100, 3))

    img_aray = tf.keras.preprocessing.image.img_to_array(img)

    preprocessed_img = tf.keras.applications.resnet50.preprocess_input(img_aray)

    preprocessed_img = tf.convert_to_tensor(preprocessed_img, tf.float32)

    return preprocessed_img, label





def prepare_label(image, label):

    label = label.numpy()

    label = label.decode('utf-8')

    logger.debug('converting label {} as one hot vector'.format(label))

    logger.debug('index of label {} is {}'.format(label, unique_labels.index(label)))

    one_hot_label = tf.one_hot(indices=unique_labels.index(label), depth=total_unique_labels)

    return image, one_hot_label



def data_pipeline(images, labels):

    logger.info(images)

    shuffle_size = len(labels)

    dataset = (

        tf.data.Dataset.from_tensor_slices((images, labels))

        .shuffle(shuffle_size)

        .map(

            lambda imageFileName, label: tf.py_function(

                prepare_image,

                [imageFileName, label],

                (tf.float32, tf.dtypes.string)

            ),

            num_parallel_calls=NUM_PARALLEL_CALLS

        )

        .map(

            lambda img, label: tf.py_function(

                prepare_label,

                [img, label],

                (tf.float32, tf.float32)

            ),

            num_parallel_calls=NUM_PARALLEL_CALLS

        )

        .repeat()

        .batch(BATCH_SIZE)

        .prefetch(1)

    )

     # Create reinitializable iterator from dataset

    iterator = dataset.make_one_shot_iterator()

    

    return iterator
# Model creation

def prepare_model() -> tf.keras.Model:

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (7, 7), activation='elu', input_shape=(100, 100, 3)))

    model.add(tf.keras.layers.BatchNormalization(axis=3))

    model.add(tf.keras.layers.Activation('elu'))

    model.add(tf.keras.layers.MaxPool2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='elu'))

    model.add(tf.keras.layers.AveragePooling2D((3, 3)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(500, activation='elu'))

    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(total_unique_labels, activation='softmax'))

    logger.debug(model.output_shape)

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    return model
imageFilenames = train_df['Image']

imageFilenames = [BASE_DIR + TRAIN_DIR + imageFileName for imageFileName in imageFilenames]

labels = tf.convert_to_tensor(train_df['Id'], dtype=tf.dtypes.string)

imageFilenames = tf.convert_to_tensor(imageFilenames, dtype=tf.dtypes.string)

iterator = data_pipeline(imageFilenames, labels)
model = prepare_model()

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='./weights.hdf5', verbose=1, save_best_only=True)

history = model.fit(iterator, steps_per_epoch=5, epochs=100, verbose=1, callbacks=[checkpointer])
plt.plot(history.history['acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()
plt.plot(history.history['loss'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()
history.history
org_test_files = os.listdir(BASE_DIR + TEST_DIR)

test_files = [BASE_DIR + TEST_DIR + file for file in org_test_files]

test_image = random.choice(test_files)

Image(filename=test_image)
output_file = open('output.csv','w')

column= ['Image', 'Id']

wrtr = csv.writer(output_file, delimiter=',')

wrtr.writerow(column)

for i, v in enumerate(test_files):

    v = tf.convert_to_tensor(v)

    preproccesed_img, label = prepare_image(v, v)

    preproccesed_img = preproccesed_img.numpy()

    preproccesed_img = np.reshape(preproccesed_img, (1, 100, 100, 3))

    print(preproccesed_img.shape)

    predicted_value = model.predict(np.array(preproccesed_img), verbose=1)

    predicted_value = predicted_value.argpartition(-4)[0][-4:]

    print(predicted_value)

    predictions = [unique_labels[p] for p in predicted_value]

    result = [org_test_files[i], ' '.join(predictions)]

    print(result)

    wrtr.writerow(result)

    

output_file.close()

    