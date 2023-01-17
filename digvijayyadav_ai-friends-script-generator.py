# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Flatten, MaxPooling2D

from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

import helper
data_dir = tf.keras.utils.get_file('Friends_Transcript.txt', 'https://raw.githubusercontent.com/uragirii/Friends-Generator/master/Data/Friends_Transcript.txt')

with open(data_dir) as f:

    text = f.readlines()

text = [x.strip() for x in text] 

print(text[:20])
text = open(data_dir, 'rb').read().decode(encoding='utf-8')

# length of text is the number of characters in it

print ('Length of text: {} characters'.format(len(text)))
#Vocablary can be used to get the unique texts in the dataset

vocab = sorted(set(text))

print('The Number of Unique words : {}'.format(len(vocab)))
char_to_idx = {u:i for i, u in enumerate(vocab)}

idx_to_char = np.array(vocab)



text_as_int = np.array([char_to_idx[c] for c in text])
print('{')

for char,_ in zip(char_to_idx, range(25)):

    print('  {:4s}: {:3d},'.format(repr(char), char_to_idx[char]))

print('  ...\n}')
# Show how the first 13 characters from the text are mapped to integers

print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))
#Maximum length of sequence to be taken as a single input in characters

seq_length = 100

examples_per_epochs = len(text)//(seq_length + 1)



# Create training examples / targets

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)



for i in char_dataset.take(5):

    print(idx_to_char[i.numpy()])
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)



for item in sequences.take(5):

    print(repr(''.join(idx_to_char[item.numpy()])))
def split_input_target(chunk):

    input_text = chunk[:-1]

    target_text = chunk[1:]

    return input_text, target_text



dataset = sequences.map(split_input_target)
for input_example, target_example in  dataset.take(1):

    print ('Input data: ', repr(''.join(idx_to_char[input_example.numpy()])))

    print ('Target data:', repr(''.join(idx_to_char[target_example.numpy()])))
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):

    print("Step {:4d}".format(i))

    print("  input: {} ({:s})".format(input_idx, repr(idx_to_char[input_idx])))

    print("  expected output: {} ({:s})".format(target_idx, repr(idx_to_char[target_idx])))
# Batch size

BATCH_SIZE = 64



# Buffer size to shuffle the dataset

# (TF data is designed to work with possibly infinite sequences,

# so it doesn't attempt to shuffle the entire sequence in memory. Instead,

# it maintains a buffer in which it shuffles elements).

BUFFER_SIZE = 10000



dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)



dataset
# Length of the vocabulary in chars

vocab_size = len(vocab)



# The embedding dimension

embedding_dim = 256



# Number of RNN units

rnn_units = 1024
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):

    model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim,

                              batch_input_shape=[batch_size, None]),

    tf.keras.layers.GRU(rnn_units,

                        return_sequences=True,

                        stateful=True,

                        recurrent_initializer='glorot_uniform'),

    tf.keras.layers.Dense(vocab_size)

  ])

    return model
model = build_model(

  vocab_size = len(vocab),

  embedding_dim=embedding_dim,

  rnn_units=rnn_units,

  batch_size=BATCH_SIZE)
model.summary()
for input_example_batch, target_example_batch in dataset.take(1):

    example_batch_predictions = model(input_example_batch)

    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)

sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
sampled_indices
print("Input: \n", repr("".join(idx_to_char[input_example_batch[0]])))

print()

print("Next Char Predictions: \n", repr("".join(idx_to_char[sampled_indices ])))
def loss(labels, logits):

    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True )

model.compile(optimizer = 'adam', loss = loss)
#saving the model

checkpoint_dir = '../training_checkpoints'

# Name of the checkpoint files

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")



checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(

    filepath=checkpoint_prefix,

    save_weights_only=True)
import shutil

shutil.rmtree(checkpoint_dir)
epochs = 30

history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])
tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()
plt.figure(figsize=(5,5))

plt.plot(history.history['loss'], label='Loss')



plt.legend()

plt.title('Metrics estimations')
def generate_text(model, start_string):

  # Evaluation step (generating text using the learned model)



  # Number of characters to generate

    num_generate = 1000



  # Converting our start string to numbers (vectorizing)

    input_eval = [char_to_idx[s] for s in start_string]

    input_eval = tf.expand_dims(input_eval, 0)



  # Empty string to store our results

    text_generated = []



  # Low temperatures results in more predictable text.

  # Higher temperatures results in more surprising text.

  # Experiment to find the best setting.

    temperature = 1.0



  # Here batch size == 1

    model.reset_states()

    for i in range(num_generate):

        predictions = model(input_eval)

      # remove the batch dimension

        predictions = tf.squeeze(predictions, 0)



      # using a categorical distribution to predict the character returned by the model

        predictions = predictions / temperature

        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()



      # We pass the predicted character as the next input to the model

      # along with the previous hidden state

        input_eval = tf.expand_dims([predicted_id], 0)



        text_generated.append(idx_to_char[predicted_id])



    return (start_string + ''.join(text_generated))
print(generate_text(model, start_string=u"Chandler: "))