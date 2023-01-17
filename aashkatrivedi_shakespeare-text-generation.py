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
import numpy as np
import pandas as pd
import nltk
import os
import time
#check if decoding is needed: text may need to be decoded as utf-8
text = open('/kaggle/input/shakespeare-text/text.txt', 'r').read() 
print(text[:200])
#Find Vocabulary (set of characters)
vocabulary = sorted(set(text))
print('No. of unique characters: {}'.format(len(vocabulary)))
#character to index mapping
char2index = {c:i for i,c in enumerate(vocabulary)}
int_text = np.array([char2index[i] for i in text])

#Index to character mapping
index2char = np.array(vocabulary)
#Testing
print("Character to Index: \n")
for char,_ in zip(char2index, range(65)):
    print('  {:4s}: {:3d}'.format(repr(char), char2index[char]))

print("\nInput text to Integer: \n")
print('{} mapped to {}'.format(repr(text[:20]),int_text[:20])) #use repr() for debugging
seq_length= 150 #max number of characters that can be fed as a single input
examples_per_epoch = len(text)

#converts text (vector) into character index stream
#Reference: https://www.tensorflow.org/api_docs/python/tf/data/Dataset
char_dataset = tf.data.Dataset.from_tensor_slices(int_text)
#Create sequences from the individual characters. Our required size will be seq_length + 1 (character RNN)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
#Testing
print("Character Stream: \n")
for i in char_dataset.take(10):
  print(index2char[i.numpy()])  

print("\nSequence: \n")
for i in sequences.take(10):
  print(repr(''.join(index2char[i.numpy()])))  #use repr() for more clarity. str() keeps formatting it
def create_input_target_pair(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(create_input_target_pair)
#Testing
for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(index2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(index2char[target_example.numpy()])))
#Creating batches

BATCH_SIZE = 64

# Buffer used to shuffle the dataset 
# Reference: https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset
vocab_size = len(vocabulary)
embedding_dim = 256
rnn_units= 1024
def build_model_lstm(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units, 
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
    return model

# Reference for theory: https://jhui.github.io/2017/03/15/RNN-LSTM-GRU/
lstm_model = build_model_lstm(
  vocab_size = vocab_size,
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)
#Testing: shape
for input_example_batch, target_example_batch in dataset.take(1):
    example_prediction = lstm_model(input_example_batch)
    assert (example_prediction.shape == (BATCH_SIZE, seq_length, vocab_size)), "Shape error"
    #print(example_prediction.shape)
#model.summary() 
#check shapes if necessary
sampled_indices = tf.random.categorical(example_prediction[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

#Loss Function reference: https://www.dlology.com/blog/how-to-use-keras-sparse_categorical_crossentropy/

example_loss  = loss(target_example_batch, example_prediction)
print("Prediction shape: ", example_prediction.shape)
print("Loss:      ", example_loss.numpy().mean())
lstm_model.compile(optimizer='adam', loss=loss)
lstm_dir_checkpoints= './training_checkpoints_LSTM'
checkpoint_prefix = os.path.join(lstm_dir_checkpoints, "checkpt_{epoch}") #name
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True)
EPOCHS=60 #increase number of epochs for better results (lesser loss)
history = lstm_model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
tf.train.latest_checkpoint(lstm_dir_checkpoints)
lstm_model = build_model_lstm(vocab_size, embedding_dim, rnn_units, batch_size=1)
lstm_model.load_weights(tf.train.latest_checkpoint(lstm_dir_checkpoints))
lstm_model.build(tf.TensorShape([1, None]))

lstm_model.summary()
def generate_text(model, start_string):
    num_generate = 1000 #Number of characters to be generated

    input_eval = [char2index[s] for s in start_string] #vectorising input
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 0.5

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
#Testing
#print(generate_text(lstm_model, start_string=u"ROMEO: "))
#Prediction with User Input
lstm_test = input("Enter your starting string: ")
print(generate_text(lstm_model, start_string=lstm_test))

