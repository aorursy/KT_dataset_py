from __future__ import absolute_import, division, print_function



import tensorflow as tf

tf.enable_eager_execution()



import numpy as np

import os

import time
path_to_file = '../input/myText.txt'
# Read, then decode for py2 compat.

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# length of text is the number of characters in it

print ('Length of text: {} characters'.format(len(text)))
# Take a look at the first 250 characters in text

print(text[:250])
# The unique characters in the file

vocab = sorted(set(text))

print ('{} unique characters'.format(len(vocab)))
# Creating a mapping from unique characters to indices

char2idx = {u:i for i, u in enumerate(vocab)}

idx2char = np.array(vocab)



text_as_int = np.array([char2idx[c] for c in text])
print('{')

for char,_ in zip(char2idx, range(20)):

    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))

print('  ...\n}')
# Show how the first 13 characters from the text are mapped to integers

print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))
# The maximum length sentence we want for a single input in characters

seq_length = 100

examples_per_epoch = len(text)//seq_length



# Create training examples / targets

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)



for i in char_dataset.take(10):

    print(idx2char[i.numpy()])
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
# Batch size 

BATCH_SIZE = 64

steps_per_epoch = examples_per_epoch//BATCH_SIZE



BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset
vocab_size = len(vocab)

embedding_dim = 256

rnn_units = 1024
rnn = tf.keras.layers.CuDNNGRU
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):

    model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, 

                              batch_input_shape=[batch_size, None]),

    rnn(rnn_units,

        return_sequences=True, 

        recurrent_initializer='glorot_uniform',

        stateful=True),

    tf.keras.layers.Dense(vocab_size)

  ])

    return model
model = build_model(

  vocab_size = len(vocab), 

  embedding_dim=embedding_dim, 

  rnn_units=rnn_units, 

  batch_size=BATCH_SIZE)
for input_example_batch, target_example_batch in dataset.take(1):

    example_batch_predictions = model(input_example_batch)

    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
model.summary()
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)

sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
# Train the Model



def loss(labels, logits):

    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)



example_batch_loss  = loss(target_example_batch, example_batch_predictions)

print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)") 

print("scalar_loss:      ", example_batch_loss.numpy().mean())
model.compile(

    optimizer = tf.train.AdamOptimizer(),

    loss = loss)
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

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()
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

        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

      

        text_generated.append(idx2char[predicted_id])



    return (start_string + ''.join(text_generated))



print(generate_text(model, start_string=u"AQUAMAN: "))