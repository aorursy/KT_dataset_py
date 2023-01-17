# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

import tensorflow_hub as hub

import tensorflow_datasets as tfds
train_data, validation_data, test_data = tfds.load(

    name="imdb_reviews", 

    split=('train[:60%]', 'train[60%:]', 'test'),

    as_supervised=True)
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

train_examples_batch
train_labels_batch
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"

hub_layer = hub.KerasLayer(embedding, input_shape=[], 

                           dtype=tf.string, trainable=True)

hub_layer(train_examples_batch[:3])
model = tf.keras.Sequential()

model.add(hub_layer)

model.add(tf.keras.layers.Dense(16, activation='relu'))

model.add(tf.keras.layers.Dense(1))



model.summary()
model.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])
history = model.fit(train_data.shuffle(10000).batch(512),

                    epochs=25,

                    validation_data=validation_data.batch(512),

                    verbose=1)
results = model.evaluate(test_data.batch(512), verbose=2)



for name, value in zip(model.metrics_names, results):

  print("%s: %.3f" % (name, value))
import matplotlib.pyplot as plt



def plotmodelhistory(history): 

    fig, axs = plt.subplots(1,2,figsize=(15,5)) 

    # summarize history for accuracy

    axs[0].plot(history.history['accuracy']) 

    axs[0].plot(history.history['val_accuracy']) 

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy') 

    axs[0].set_xlabel('Epoch')

    axs[0].legend(['train', 'validate'], loc='upper left')

    # summarize history for loss

    axs[1].plot(history.history['loss']) 

    axs[1].plot(history.history['val_loss']) 

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss') 

    axs[1].set_xlabel('Epoch')

    axs[1].legend(['train', 'validate'], loc='upper left')

    plt.show()



# list all data in history

print(history.history.keys())



plotmodelhistory(history)