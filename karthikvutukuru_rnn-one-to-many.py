# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from __future__ import absolute_import, division, print_function, unicode_literals



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

import os 

import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

path_to_file
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

print('Length of text: {} characters'.format(len(text)))
print(text[:250])
# Unique characters in file

vocab = sorted(set(text))

print ('{} unique characters'.format(len(vocab)))

# Text processing

char2idx = {u : i for i,u in enumerate(vocab)}

idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])
print('{')

for char, _ in zip(char2idx, range(20)):

    print(' {:4s}: {:3d},'.format(repr(char), char2idx[char]))

print(' ...\n')

    
print('{} --- characters mapped to int --- > {}'.format(repr(text[:13]), text_as_int[:13]))
seq_length = 120

examples_per_epoch = len(text)//(seq_length+1)



# Create Dataset

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)



for i in char_dataset.take(5):

    print(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):

    print(repr(''.join(idx2char[item.numpy()])))
# for each sequence, duplicate and shift it to form the input and target



def split_input_target(chunk):

    input_text = chunk[:-1]

    target_text = chunk[1:]

    return input_text, target_text



dataset = sequences.map(split_input_target)



    
for input_example, target_example in dataset.take(1):

    print('Input Data:', repr(''.join(idx2char[input_example.numpy()])))

    print('Target Data:', repr(''.join(idx2char[target_example.numpy()])))
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):

    print('Step {:4d}'.format(i))

    print('input: {} ({:s})'.format(input_idx, repr(idx2char[input_idx])))

    print('target: {} ({:s})'.format(target_idx, repr(idx2char[target_idx])))

# Create Training Batches

BATCH_SIZE = 64

BUFFER_SIZE = 1000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset
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
model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

model
model.summary()
for input_example_batch, target_example_batch in dataset.take(1):

    example_batch_predictions = model(input_example_batch)

    print(example_batch_predictions.shape, '# (batch_size, sequence_length, vocab_size)')

    
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)

sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

sampled_indices
# Decode the above to see the text

print('Input: \n', repr(''.join(idx2char[input_example_batch[0]])))

print()

print('Output: \n',repr(''.join(idx2char[sampled_indices])))
def loss(labels, logits):

    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)



example_batch_loss = loss(target_example_batch, example_batch_predictions)

print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")

print("scalar_loss:      ", example_batch_loss.numpy().mean())
model.compile(optimizer='adam', loss=loss)
# Configure checkpoints



checkpoint_dir = './training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(

    filepath=checkpoint_prefix,

    save_weights_only=True)
EPOCHS=10

history=model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
# Generate Text



tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))
model.summary()
def generate_text(model, start_string):

    

    # no. of characters to generate

    num_generate = 1000

    

    # Convert start string to numbers

    input_eval = [char2idx[s] for s in start_string]

    input_eval = tf.expand_dims(input_eval, 0)

    

    text_generated = []

    

    temp = 1.0

    

    model.reset_states()

    for i in range(num_generate):

        predictions = model(input_eval)

        

        # remove batch dimension

        predictions = tf.squeeze(predictions, 0)

        

        # use categorical distribution to predict the character returned

        predictions = predictions/temp

        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        

        input_eval = tf.expand_dims([predicted_id], 0)

        

        text_generated.append(idx2char[predicted_id])

        

    return start_string+''.join(text_generated)

        
print(generate_text(model, start_string=u"ROMEO: "))

model = build_model(

  vocab_size = len(vocab),

  embedding_dim=embedding_dim,

  rnn_units=rnn_units,

  batch_size=BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()

@tf.function

def train_step(inp, target):

    with tf.GradientTape() as tape:

        predictions = model(inp)

        loss = tf.reduce_mean(

        tf.keras.losses.sparse_categorical_crossentropy(

        target, predictions, from_logits=True)

        )

    grads = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    

    return loss      
EPOCHS = 10



for epoch in range(EPOCHS):

    start = time.time()

    hidden = model.reset_states()

    for (batch_n, (inp, target)) in enumerate(dataset):

        loss = train_step(inp, target)

        

        if batch_n % 100 == 0:

            template = 'Epoch {} Batch {} Loss {}'

            print(template.format(epoch+1, batch_n, loss))

            

    if (epoch + 1) % 5 == 0:

        model.save_weights(checkpoint_prefix.format(epoch=epoch))

        

    print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))

    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    

model.save_weights(checkpoint_prefix.format(epoch=epoch))

model.summary()
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)



model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))



model.build(tf.TensorShape([1, None]))
print(generate_text(model, start_string=u"ROMEO: "))
