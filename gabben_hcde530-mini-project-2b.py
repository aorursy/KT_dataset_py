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
import json
from pandas.io.json import json_normalize
metadata = pd.read_json("../input/gutenberg-dammit/gutenberg-dammit-files/gutenberg-metadata.json")
print(metadata.shape)
metadata.head()
metadata.loc[1000:1050]
with open('../input/gutenberg-dammit/gutenberg-dammit-files/010/01074.txt', 'r') as text:
    data = text.read().replace('\n', ' ')
    print(data)
open_text = open("../input/gutenberg-dammit/gutenberg-dammit-files/010/01074.txt", "r")
print(open_text.read())
print ('Length of text: {} characters'.format(len(data)))
import nltk
from nltk.tokenize import sent_tokenize
tokenized_text=sent_tokenize(data)
print(tokenized_text)
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(data)
print(tokenized_word)
from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist)
#What are the two most common words?
fdist.most_common(2)
#What are the ten most common?
fdist.most_common(10)
#What are the 10 least common?
fdist.most_common()[-10:]
import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False) #show me the 30 most common words
plt.show()
fdist.plot(50,cumulative=False) #show me the 50 most common words
plt.show()
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)
filtered_sent=[]  #create an empty list for the text empty of stop words
for w in tokenized_text:     #go through every word in the text
    if w not in stop_words:  #if it's not a stop word
        filtered_sent.append(w)  #add it to the list
print("Tokenized Sentence:",tokenized_text)  #print the text
print("Filtered Sentence:",filtered_sent) #print the text without stop words
print ('Length of text: {} characters'.format(len(data))) 
print ('Length of filtered text: {} characters'.format(len(filtered_sent)))
vocab = sorted(set(data))
print ('{} unique characters'.format(len(vocab)))
#Each unique character gets assigned a number
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in data])

#print the table
print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')
#print the first 13 characters as an array
print ('{} ---- characters mapped to int ---- > {}'.format(repr(data[:13]), text_as_int[:13]))
# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(data)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# Print the first 5 characters
for i in char_dataset.take(5): 
  print(idx2char[i.numpy()])
# Turn the character set into a sequence using .batch
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# Print 10 sequences
for item in sequences.take(10):
  print(repr(''.join(idx2char[item.numpy()])))
# Input & target text mapping function
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# Apply the function to each of my sequences
dataset = sequences.map(split_input_target)

# Print one mapped sequence
for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))
# for each of the first five characters of the sequence, print out the input and the target
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset
# Length of the vocabulary in characters
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

# Build the model using 3 layers.
def build_model(vocab_size, embedding_dim, rnn_units, batch_size): #let's specify the dimensions
  model = tf.keras.Sequential([  #used to embed all three layers
    tf.keras.layers.Embedding(vocab_size, embedding_dim, #input later
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units, #RNN
                        return_sequences=True,
                        stateful=True,  #it needs tobe stateful! 
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size) #output layer
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
sampled_indices
print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))
# Function that sets the from_logits flag
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
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
EPOCHS=10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
#start training from the latest checkpoint
tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()
def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting the first string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Temperature is the 'creativity' variable:
  # Low temperatures result in more predictable text
  # Higher temperatures result in more surprising text
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

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"The")) #specifiy the first word of the text as a prompt