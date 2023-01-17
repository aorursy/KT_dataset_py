from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

stream = open('/kaggle/input/dialogues.txt' , 'r')
import re

import nltk

from keras.models import Model

from keras.layers import Input, LSTM, Dense

import numpy as np



batch_size = 64  # Batch size for training.

epochs = 100  # Number of epochs to train for.

latent_dim = 256  # Latent dimensionality of the encoding space.
input_texts = []

target_texts = []

input_characters = set()

target_characters = set()
for linenum,line in enumerate(stream):

    line = line.replace('-','').replace('\n','').strip().lower()

    if line == '\n' or line == '':

        continue

    if linenum % 2 == 0:        

        input_texts.append(line)

        for char in line:

            if char not in input_characters:

                input_characters.add(char)

    else:

        target_texts.append(line)

        for char in line:

            if char not in target_characters:

                target_characters.add(char)
input_texts = input_texts[:100000]

target_texts = target_texts[:100000]

input_characters = sorted(list(input_characters))

target_characters = sorted(list(target_characters))

num_encoder_tokens = len(input_characters)

num_decoder_tokens = len(target_characters)

max_encoder_seq_length = max([len(txt) for txt in input_texts])

max_decoder_seq_length = max([len(txt) for txt in target_texts])
print('Number of samples:', len(input_texts))

print('Number of unique input tokens:', num_encoder_tokens)

print('Number of unique output tokens:', num_decoder_tokens)

print('Max sequence length for inputs:', max_encoder_seq_length)

print('Max sequence length for outputs:', max_decoder_seq_length)
input_token_index = dict(

    [(char, i) for i, char in enumerate(input_characters)])

target_token_index = dict(

    [(char, i) for i, char in enumerate(target_characters)])



encoder_input_data = np.zeros(

    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),

    dtype='float32')

decoder_input_data = np.zeros(

    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),

    dtype='float32')

decoder_target_data = np.zeros(

    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),

    dtype='float32')
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):

    for t, char in enumerate(input_text):

        encoder_input_data[i, t, input_token_index[char]] = 1.

    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.

    for t, char in enumerate(target_text):

        # decoder_target_data is ahead of decoder_input_data by one timestep

        decoder_input_data[i, t, target_token_index[char]] = 1.

        if t > 0:

            # decoder_target_data will be ahead by one timestep

            # and will not include the start character.

            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.

    decoder_target_data[i, t:, target_token_index[' ']] = 1.
import keras
callbacks_list = [

    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=1)

]
# Define an input sequence and process it.

encoder_inputs = Input(shape=(None, num_encoder_tokens))

encoder = LSTM(latent_dim, return_state=True)

encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.

encoder_states = [state_h, state_c]



# Set up the decoder, using `encoder_states` as initial state.

decoder_inputs = Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences,

# and to return internal states as well. We don't use the

# return states in the training model, but we will use them in inference.

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(decoder_inputs,

                                     initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)



# Define the model that will turn

# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)



# # Run training

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',

              metrics=['accuracy'])

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,

          batch_size=batch_size,

          epochs=epochs,

          callbacks=callbacks_list,

          validation_split=0.2)

# Save model

model.save('s2s.h5')
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label="loss")

plt.plot(history.history['val_loss'], label="val_loss")

plt.plot(history.history['accuracy'], label="accuracy")

plt.plot(history.history['val_accuracy'], label="val_accuracy")

plt.xlabel('epoch')

plt.legend()

plt.show()
encoder_model = Model(encoder_inputs, encoder_states)



decoder_state_input_h = Input(shape=(latent_dim,))

decoder_state_input_c = Input(shape=(latent_dim,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)



# Reverse-lookup token index to decode sequences back to

# something readable.

reverse_input_char_index = dict(

    (i, char) for char, i in input_token_index.items())

reverse_target_char_index = dict(

    (i, char) for char, i in target_token_index.items())
def decode_sequence(input_seq):

    # Encode the input as state vectors.

    states_value = encoder_model.predict(input_seq)



    # Generate empty target sequence of length 1.

    target_seq = np.zeros((1, 1, num_decoder_tokens))

    # Populate the first character of target sequence with the start character.

    target_seq[0, 0, target_token_index['\t']] = 1.



    # Sampling loop for a batch of sequences

    # (to simplify, here we assume a batch of size 1).

    stop_condition = False

    decoded_sentence = ''

    while not stop_condition:

        output_tokens, h, c = decoder_model.predict(

            [target_seq] + states_value)



        # Sample a token

        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        sampled_char = reverse_target_char_index[sampled_token_index]

        decoded_sentence += sampled_char



        # Exit condition: either hit max length

        # or find stop character.

        if (sampled_char == '\n' or

           len(decoded_sentence) > max_decoder_seq_length):

            stop_condition = True



        # Update the target sequence (of length 1).

        target_seq = np.zeros((1, 1, num_decoder_tokens))

        target_seq[0, 0, sampled_token_index] = 1.



        # Update states

        states_value = [h, c]



    return decoded_sentence
# encoder_input_data1 = ["Привет"]
for seq_index in range(100):

    # Take one sequence (part of the training set)

    # for trying out decoding.

    input_seq = encoder_input_data1[seq_index: seq_index + 1]

    decoded_sentence = decode_sequence(input_seq)

    print('-')

    print('Input sentence:', input_texts[seq_index])

    print('Decoded sentence:', decoded_sentence)
import numpy as np

import tensorflow as tf

from tensorflow.keras import layers , activations , models , preprocessing , utils

import pandas as pd

stream.close()
questions
def vocab_creater(text_lists, VOCAB_SIZE):



  tokenizer = Tokenizer(num_words=VOCAB_SIZE)

  tokenizer.fit_on_texts(text_lists)

  dictionary = tokenizer.word_index

  

  word2idx = {}

  idx2word = {}

  for k, v in dictionary.items():

      if v < VOCAB_SIZE:

          word2idx[k] = v

          idx2word[v] = k

      if v >= VOCAB_SIZE-1:

          continue

          

  return word2idx, idx2word
word2idx, idx2word = vocab_creater(text_lists=questions+answers, VOCAB_SIZE=14999)

VOCAB_SIZE = 14999
tokenizer = Tokenizer(num_words=VOCAB_SIZE)

encoder_sequences = tokenizer.texts_to_sequences(answers[:20])

encoder_sequences
def text2seq(encoder_text, decoder_text, VOCAB_SIZE):



  tokenizer = Tokenizer(num_words=VOCAB_SIZE)

  encoder_sequences = tokenizer.texts_to_sequences(encoder_text)

  decoder_sequences = tokenizer.texts_to_sequences(decoder_text)

  

  return encoder_sequences, decoder_sequences



encoder_sequences, decoder_sequences = text2seq(questions, answers, VOCAB_SIZE) 
encoder_sequences
import keras
from keras.preprocessing.text import Tokenizer
stream = open('/kaggle/input/dialogues.txt' , 'r')
questions = []

answers = []
for linenum,line in enumerate(stream):

    line = line.replace('-','').replace('\n','').strip().lower()

    if line == '\n' or line == '':

        continue

    if linenum % 2 == 0:        

        questions.append(line)

#         for char in line:

#             if char not in input_characters:

#                 input_characters.add(char)

    else:

        answers.append("<start>"+line+"<end>")

#         for char in line:

#             if char not in target_characters:

#                 target_characters.add(char)
questions = questions[:1000]
answers = answers[:1000]
tokenizer = preprocessing.text.Tokenizer()

tokenizer.fit_on_texts( questions ) 

tokenized_questions_lines = tokenizer.texts_to_sequences( questions )
length_list = list()

for token_seq in tokenized_questions_lines:

    length_list.append( len( token_seq ))

max_input_length = np.array( length_list ).max()

print( 'English max length is {}'.format( max_input_length ))
padded_questions_lines = preprocessing.sequence.pad_sequences( tokenized_questions_lines , maxlen=max_input_length , padding='post' )

encoder_input_data = np.array( padded_questions_lines )

print( 'Encoder input data shape -> {}'.format( encoder_input_data.shape ))
questions_word_dict = tokenizer.word_index

num_questions_tokens = len( questions_word_dict )+1

print( 'Number of questions tokens = {}'.format( num_questions_tokens))
tokenizer = preprocessing.text.Tokenizer()

tokenizer.fit_on_texts( answers ) 

tokenized_answers_lines = tokenizer.texts_to_sequences( answers ) 



length_list = list()

for token_seq in tokenized_answers_lines:

    length_list.append( len( token_seq ))

max_output_length = np.array( length_list ).max()

print( 'Marathi max length is {}'.format( max_output_length ))



padded_answers_lines = preprocessing.sequence.pad_sequences( tokenized_answers_lines , maxlen=max_output_length, padding='post' )

decoder_input_data = np.array( padded_answers_lines )

print( 'Decoder input data shape -> {}'.format( decoder_input_data.shape ))



answers_word_dict = tokenizer.word_index

num_answers_tokens = len( answers_word_dict )+1

print( 'Number of answers tokens = {}'.format( num_answers_tokens))
decoder_target_data = list()

for token_seq in tokenized_answers_lines:

    decoder_target_data.append( token_seq[ 1 : ] ) 
padded_answers_lines = preprocessing.sequence.pad_sequences( decoder_target_data , maxlen=max_output_length, padding='post' )

onehot_answers_lines = utils.to_categorical( padded_answers_lines , num_answers_tokens )

decoder_target_data = np.array( onehot_answers_lines )

print( 'Decoder target data shape -> {}'.format( decoder_target_data.shape ))
num_encoder_tokens = len(questions)

num_decoder_tokens = len(answers)
latent_dim
encoder_inputs = tf.keras.layers.Input(shape=( None , ))

encoder_embedding = tf.keras.layers.Embedding( num_encoder_tokens, 256 , mask_zero=True ) (encoder_inputs)

encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 128 , return_state=True  )( encoder_embedding )

encoder_states = [ state_h , state_c ]



decoder_inputs = tf.keras.layers.Input(shape=( None ,  ))

decoder_embedding = tf.keras.layers.Embedding( num_decoder_tokens, 256 , mask_zero=True) (decoder_inputs)

decoder_lstm = tf.keras.layers.LSTM( 128 , return_state=True , return_sequences=True)

decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )

decoder_dense = tf.keras.layers.Dense( num_decoder_tokens , activation=tf.keras.activations.softmax ) 

output = decoder_dense ( decoder_outputs )



model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy')



model.summary()

model.fit([encoder_input_data , decoder_input_data], decoder_target_data, batch_size=250, epochs=50 ) 

# Save model

model.save('s2s.h5')
# encoder_inputs = tf.keras.layers.Input(shape=( None , ))

# encoder_embedding = tf.keras.layers.Embedding( num_questions_tokens, 256 , mask_zero=True ) (encoder_inputs)

# encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 128 , return_state=True  )( encoder_embedding )

# encoder_states = [ state_h , state_c ]



# decoder_inputs = tf.keras.layers.Input(shape=( None ,  ))

# decoder_embedding = tf.keras.layers.Embedding( num_answers_tokens, 256 , mask_zero=True) (decoder_inputs)

# decoder_lstm = tf.keras.layers.LSTM( 128 , return_state=True , return_sequences=True)

# decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )

# decoder_dense = tf.keras.layers.Dense( num_answers_tokens , activation=tf.keras.activations.softmax ) 

# output = decoder_dense ( decoder_outputs )



# model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )

# model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')



# model.summary()
# history = model.fit([encoder_input_data , decoder_input_data], decoder_target_data, batch_size=250, epochs=1 ) 

history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,

          batch_size=32,

          epochs=1,

#           callbacks=callbacks_list,

          validation_split=0.2)

model.save( 'model.h5' ) 