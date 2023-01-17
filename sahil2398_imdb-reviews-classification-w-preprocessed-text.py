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

from tensorflow import keras



import tensorflow_datasets as tfds

tfds.disable_progress_bar()
(train_data, test_data), info = tfds.load(

    # Use the version pre-encoded with an ~8k vocabulary.

    'imdb_reviews/subwords8k', 

    # Return the train/test datasets as a tuple.

    split = (tfds.Split.TRAIN, tfds.Split.TEST),

    # Return (example, label) pairs from the dataset (instead of a dictionary).

    as_supervised=True,

    # Also return the `info` structure. 

    with_info=True)
encoder = info.features['text'].encoder
print ('Vocabulary size: {}'.format(encoder.vocab_size))
sample_string = 'Hello TensorFlow.'



encoded_string = encoder.encode(sample_string)

print ('Encoded string is {}'.format(encoded_string))



original_string = encoder.decode(encoded_string)

print ('The original string: "{}"'.format(original_string))



assert original_string == sample_string
for ts in encoded_string:

  print ('{} ----> {}'.format(ts, encoder.decode([ts])))
for train_example, train_label in train_data.take(1):

  print('Encoded text:', train_example[:10].numpy())

  print('Label:', train_label.numpy())
BUFFER_SIZE = 1000



train_batches = (

    train_data

    .shuffle(BUFFER_SIZE)

    .padded_batch(32))



test_batches = (

    test_data

    .padded_batch(32))
for example_batch, label_batch in train_batches.take(2):

  print("Batch shape:", example_batch.shape)

  print("label shape:", label_batch.shape)

  
model = keras.Sequential([

  keras.layers.Embedding(encoder.vocab_size, 16),

  keras.layers.GlobalAveragePooling1D(),

  keras.layers.Dense(1)])



model.summary()
model.compile(optimizer='adam',

              loss=tf.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])
history = model.fit(train_batches,

                    epochs=30,

                    validation_data=test_batches,

                    validation_steps=30)
loss, accuracy = model.evaluate(test_batches)



print("Loss: ", loss)

print("Accuracy: ", accuracy)
history_dict = history.history

history_dict.keys()
import matplotlib.pyplot as plt



acc = history_dict['accuracy']

val_acc = history_dict['val_accuracy']

loss = history_dict['loss']

val_loss = history_dict['val_loss']



epochs = range(1, len(acc) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, loss, 'bo', label='Training loss')

# b is for "solid blue line"

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()   # clear figure



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(loc='lower right')



plt.show()