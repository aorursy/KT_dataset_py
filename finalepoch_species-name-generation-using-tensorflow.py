!wget https://www.uniprot.org/docs/speclist

!cat speclist | grep ': N=' | sed 's/^[^=]*=//g' > species.txt
import random

import tensorflow as tf

import numpy as np

import os

import time



with open('species.txt') as f:

    lines = f.readlines()

    random.shuffle(lines)
text = ' '.join(lines)

vocab = sorted(set(text))

char2idx = {u:i for i, u in enumerate(vocab)}

idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])
seq_length = 100

examples_per_epoch = len(text)//(seq_length+1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)



for item in sequences.take(5):

  print(repr(''.join(idx2char[item.numpy()])))
def split_input_target(chunk):

  input_text = chunk[:-1]

  target_text = chunk[1:]

  return input_text, target_text
dataset = sequences.map(split_input_target)
for input_example, target_example in  dataset.take(1):

  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))

  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))
BATCH_SIZE = 64

BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

vocab_size = len(vocab)

embedding_dim = 256

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

print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))

print()

print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))
def loss(labels, logits):

  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
model.compile(optimizer='adam', loss=loss)

checkpoint_dir = './training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")



checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(

    filepath=checkpoint_prefix,

    save_weights_only=True)
EPOCHS=10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
tf.train.latest_checkpoint(checkpoint_dir)

p_model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

p_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

p_model.build(tf.TensorShape([1, None]))
def generate_text(model, start_string):

  num_generate = 1000

  input_eval = [char2idx[s] for s in start_string]

  input_eval = tf.expand_dims(input_eval, 0)

  text_generated = []

  temperature = 1.0  

  model.reset_states()

  for i in range(num_generate):

    predictions = model(input_eval)

    predictions = tf.squeeze(predictions, 0)

    predictions = predictions / temperature

    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(idx2char[predicted_id])



  return (start_string + ''.join(text_generated))
generate_text(p_model,'ted')