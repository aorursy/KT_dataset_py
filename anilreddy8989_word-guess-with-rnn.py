# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from __future__ import absolute_import, division, print_function



import tensorflow as tf

tf.enable_eager_execution()

import numpy as np

import os

import time
# Adding file path

file_path = '../input/textfile.txt'



# Reading the file data, and then decoding to py2 compat.

text_data = open(file_path, 'rb').read().decode(encoding='utf-8')

# The unique characters in the file

vocab_unique = sorted(set(text_data))



# Unique characters to Indices Mapping

char2ind = {u:i for i, u in enumerate(vocab_unique)}

ind2char = np.array(vocab_unique)



text_to_int = np.array([char2ind[c] for c in text_data])



# The maximum length sentence we want for a single input in characters

seq_len = 50

text_for_epoch = len(text_data)//seq_len



# Creating training targets

target_dataset = tf.data.Dataset.from_tensor_slices(text_to_int)



for i in target_dataset.take(5):

    print(ind2char[i.numpy()])

    
sequence = target_dataset.batch(seq_len+1, drop_remainder=True)



def splitting_target_input(chunk):

    text_input = chunk[:-1]

    text_target = chunk[1:]

    return text_input, text_target



dataset = sequence.map(splitting_target_input)
for input_example, target_example in  dataset.take(1):

    print ('Input data: ', repr(''.join(ind2char[input_example.numpy()])))

    print ('Target data:', repr(''.join(ind2char[target_example.numpy()])))
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):

    print("Step {:4d}".format(i))

    print("  input: {} ({:s})".format(input_idx, repr(ind2char[input_idx])))

    print("  expected output: {} ({:s})".format(target_idx, repr(ind2char[target_idx])))
# Batch size 

BATCH_SIZE = 64

steps_per_epoch = text_for_epoch//BATCH_SIZE

BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

vocab_size = len(vocab_unique)

embedding_dim = 256

rnn_units = 1024

rnn = tf.keras.layers.CuDNNGRU 



model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, 

                              batch_input_shape=[BATCH_SIZE, None]),

    rnn(rnn_units,

        return_sequences=True, 

        recurrent_initializer='glorot_uniform',

        stateful=True),

    tf.keras.layers.Dense(vocab_size)

  ])



model.summary()
for input_example_batch, target_example_batch in dataset.take(1):

    example_batch_predictions = model(input_example_batch)
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)

sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

# Train the Model



def loss_func(labels, logits):

    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)



example_batch_loss  = loss_func(target_example_batch, example_batch_predictions)



model.compile(

    optimizer = tf.train.AdamOptimizer(),

    loss = loss_func)
# Directory where the checkpoints will be saved

checkpoint_dir = './training_checkpoints'

# Name of the checkpoint files

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")



checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(

    filepath=checkpoint_prefix,

    save_weights_only=True)
EPOCHS=10



history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])
tf.train.latest_checkpoint(checkpoint_dir)





model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, 

                              batch_input_shape=[1, None]),

    rnn(rnn_units,

        return_sequences=True, 

        recurrent_initializer='glorot_uniform',

        stateful=True),

    tf.keras.layers.Dense(vocab_size)

  ])







model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))



model.build(tf.TensorShape([1, None]))





model.summary()
# Number of characters to generate

max_words = 350



# Starting String

start_string=u"ANTONIO: "







input_eval = [char2ind[s] for s in start_string]

input_eval = tf.expand_dims(input_eval, 0)





gen_text = []



temperature = 1.0





model.reset_states()

for i in range(max_words):

    predictions = model(input_eval)



    predictions = tf.squeeze(predictions, 0)





    predictions = predictions / temperature

    predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()



  

    input_eval = tf.expand_dims([predicted_id], 0)



    gen_text.append(ind2char[predicted_id])

    

print(start_string + ''.join(gen_text))