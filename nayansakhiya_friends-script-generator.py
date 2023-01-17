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
import tensorflow as tf
friends = '/kaggle/input/friends-tv-show-script/Friends_Transcript.txt'
# Read, then decode for py2 compat.



text = open(friends, 'rb').read().decode(encoding='utf-8')



# length of text is the number of characters in it



print ('Length of text: {} characters'.format(len(text)))
# Take a look at the first 250 characters in text

print(text[:250])
# The unique characters in the file



vocab = sorted(set(text))

print ('{} unique characters'.format(len(vocab)))
# process the text



char2index = {u:i for i, u in enumerate(vocab)}

index2char = np.array(vocab)

text_as_int = np.array([char2index[c] for c in text])
print('{')

for char,_ in zip(char2index, range(20)):

    print('  {:4s}: {:3d},'.format(repr(char), char2index[char]))

print('  ...\n}')
# Show how the first 13 characters from the text are mapped to integers



print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))
# convert text vector to charcter indices



# The maximum length sentence we want for a single input in characters

seq_length = 100

examples_per_epoch = len(text)//(seq_length+1)



# Create training examples / targets

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)



for i in char_dataset.take(5):

      print(index2char[i.numpy()])
# convert character to sequence



sequences = char_dataset.batch(seq_length+1, drop_remainder=True)



for item in sequences.take(5):

    print(repr(''.join(index2char[item.numpy()])))
def split_input_target(chunk):

    input_text = chunk[:-1]

    target_text = chunk[1:]

    return input_text, target_text



dataset = sequences.map(split_input_target)
# print input and target values



for input_example, target_example in  dataset.take(1):

    print ('Input data: ', repr(''.join(index2char[input_example.numpy()])))

    print ('Target data:', repr(''.join(index2char[target_example.numpy()])))
# Batch size

BATCH_SIZE = 64

BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset
# Length of the vocabulary in chars

vocab_size = len(vocab)



# The embedding dimension

embedding_dim = 128



# Number of RNN units

rnn_units = 256
# build the model



model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim,

                              batch_input_shape=[BATCH_SIZE, None]),

    tf.keras.layers.GRU(rnn_units,

                        return_sequences=True,

                        stateful=True,

                        recurrent_initializer='glorot_uniform'),

    tf.keras.layers.Dense(vocab_size)

])

model.summary()
for input_example_batch, target_example_batch in dataset.take(1):

    example_batch_predictions = model(input_example_batch)

    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
def loss(labels, logits):

    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)



example_batch_loss  = loss(target_example_batch, example_batch_predictions)

print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")

print("scalar_loss:      ", example_batch_loss.numpy().mean())
model.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved

checkpoint_dir = './training_checkpoints'



# Name of the checkpoint files

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")



checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(

    filepath=checkpoint_prefix,

    save_weights_only=True)
history = model.fit(dataset, epochs=10, callbacks=[checkpoint_callback])
# to keep prediction step simple we use batch size of 1. for that we need to rebuild model

# and restore weight from checkpoint



tf.train.latest_checkpoint(checkpoint_dir)
# rebuild the model



model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim,

                              batch_input_shape=[1, None]),

    tf.keras.layers.GRU(rnn_units,

                        return_sequences=True,

                        stateful=True,

                        recurrent_initializer='glorot_uniform'),

    tf.keras.layers.Dense(vocab_size)

])

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()
def generate_text(model, start_string):

    # Number of characters to generate

    num_generate = 5000



    # Converting our start string to numbers (vectorizing)

    input_eval = [char2index[s] for s in start_string]

    input_eval = tf.expand_dims(input_eval, 0)



    # Empty string to store our results

    text_generated = []



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



        text_generated.append(index2char[predicted_id])



    return (start_string + ''.join(text_generated))
print(generate_text(model, start_string=u"Monica: "))
print(generate_text(model, start_string=u"Rachel: "))