import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import os, sys

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Model

from keras.layers import (Input,LSTM,GRU,Dense,

                          Embedding,Bidirectional,RepeatVector,

                          Concatenate,Activation,Dot,Lambda)

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import keras.backend as K
# function to do softmax over time

def softmax_over_time(x):

  assert(K.ndim(x) > 2)

  e = K.exp(x - K.max(x, axis=1, keepdims=True))

  s = K.sum(e, axis=1, keepdims=True)

  return e/s
# config variables

BATCH_SIZE = 64

EPOCHS = 100

LATENT_DIM = 256

LATENT_DIM_DECODER = 256

NUM_SAMPLES = 10000

MAX_SEQUENCE_LENGTH = 100

MAX_NUM_WORDS = 20000

EMBEDDING_DIM = 100
input_texts = [] # sentence in original language

target_texts = [] # sentence in target language

target_texts_inputs = [] # sentence in target language offset by 1
# load in the data

t = 0

for line in open('/kaggle/input/twitter-auto/twitter_tab_format.txt'):

  t+=1

  if t>NUM_SAMPLES:

    break

  # input and target are seperated by '\t'

  if '\t' not in line:

    continue

  input_text, translation = line.split('\t')

  target_text = translation + ' <eos>'

  target_text_input = '<sos> ' + translation

  input_texts.append(input_text)

  target_texts.append(target_text)

  target_texts_inputs.append(target_text_input)

print('num samples:',len(input_texts))
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)

tokenizer_inputs.fit_on_texts(input_texts)

input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
# word to index mapping for input language

word2idx_inputs = tokenizer_inputs.word_index

print('Found %s unique input tokens.'%len(word2idx_inputs))
# max length input seq

max_len_input = max(len(s) for s in input_sequences)
# tokenize the outputs

tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')

tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)

target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)

target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)
# word to index mapping for output language

word2idx_outputs = tokenizer_outputs.word_index

print('Found %s unique output tokens.'%len(word2idx_outputs))
num_words_output = len(word2idx_outputs)+1

# max length output seq

max_len_target = max(len(s) for s in target_sequences)
# padding the sequences

encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)

print('Encoder data shape:',encoder_inputs.shape)



decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')

print('Decoder data shape:',decoder_inputs.shape)



decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')
# loading pre-trained word vectors

word2vec = {}

with open('/kaggle/input/glove6b100dtxt/glove.6B.%sd.txt'%EMBEDDING_DIM) as f:

  for line in f:

    values = line.split()

    word = values[0]

    vec = np.asarray(values[1:],dtype='float32')

    word2vec[word] = vec

print('Found %s word vectors.'%len(word2vec))
# prepare embedding matrix

num_words = min(MAX_NUM_WORDS, len(word2idx_inputs)+1)

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word2idx_inputs.items():

  if i < MAX_NUM_WORDS:

    embedding_vector = word2vec.get(word)

    if embedding_vector is not None:

      embedding_matrix[i] = embedding_vector
# creating embedding layer

embedding_layer = Embedding(num_words,

                            EMBEDDING_DIM,

                            weights=[embedding_matrix],

                            input_length=max_len_input,

                            trainable=False)
decoder_targets_one_hot = np.zeros((len(input_texts),

                                   max_len_target,

                                   num_words_output),dtype='float32')

for i,d in enumerate(decoder_targets):

  for t,word in enumerate(d):

    decoder_targets_one_hot[i, t, word] = 1
# build the model

# setup the encoder

encoder_inputs_placeholder = Input(shape=(max_len_input,))

x = embedding_layer(encoder_inputs_placeholder)

encoder = Bidirectional(LSTM(LATENT_DIM, return_sequences=True, dropout=0.5))

encoder_outputs = encoder(x)
# setup the decoder

decoder_inputs_placeholder = Input(shape=(max_len_target,))

decoder_embedding = Embedding(num_words_output, EMBEDDING_DIM)

decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
# Attention layer

attn_repeat_layer = RepeatVector(max_len_input)

attn_concat_layer = Concatenate(axis=-1)

attn_dense1 = Dense(10, activation='tanh')

attn_dense2 = Dense(1, activation=softmax_over_time)

attn_dot = Dot(axes=1)
def one_step_attention(h, st_1):

  # h = h(1), ..., h(Tx), shape = (Tx, LATENT_DIM * 2)

  # st_1 = s(t-1), shape = (LATENT_DIM_DECODER,)

 

  # copy s(t-1) Tx times

  # now shape = (Tx, LATENT_DIM_DECODER)

  st_1 = attn_repeat_layer(st_1)



  # Concatenate all h(t)'s with s(t-1)

  # Now of shape (Tx, LATENT_DIM_DECODER + LATENT_DIM * 2)

  x = attn_concat_layer([h, st_1])



  # Neural net first layer

  x = attn_dense1(x)



  # Neural net second layer with special softmax over time

  alphas = attn_dense2(x)



  # "Dot" the alphas and the h's

  # Remember a.dot(b) = sum over a[t] * b[t]

  context = attn_dot([alphas, h])



  return context
# decoder after attention

decoder_lstm = LSTM(LATENT_DIM_DECODER, return_state=True)

decoder_dense = Dense(num_words_output, activation='softmax')
initial_s = Input(shape=(LATENT_DIM_DECODER,), name='s0')

initial_c = Input(shape=(LATENT_DIM_DECODER,), name='c0')

context_last_word_concat_layer = Concatenate(axis=2)
# s, c will be re-assigned in each iteration of the loop

s = initial_s

c = initial_c



# collect outputs in a list at first

outputs = []

for t in range(max_len_target): # Ty times

  # get the context using attention

  context = one_step_attention(encoder_outputs, s)



  # we need a different layer for each time step

  selector = Lambda(lambda x: x[:, t:t+1])

  xt = selector(decoder_inputs_x)

  

  # combine 

  decoder_lstm_input = context_last_word_concat_layer([context, xt])



  # pass the combined [context, last word] into the LSTM

  # along with [s, c]

  # get the new [s, c] and output

  o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])



  # final dense layer to get next word prediction

  decoder_outputs = decoder_dense(o)

  outputs.append(decoder_outputs)
# 'outputs' is now a list of length Ty

# each element is of shape [batch_size, output_vocab_size]

# therefore if we simply stack all the outputs into 1 tensor

# it would be of shape TxNxD

# we would like it to be of shape NxTxD



# so we stack and transpose

def stack_and_transpose(x):

  x = K.stack(x)

  x = K.permute_dimensions(x, pattern=(1, 0, 2))

  return x
# making it a layer

stacker = Lambda(stack_and_transpose)

outputs = stacker(outputs)
# create the model

model = Model(inputs=[encoder_inputs_placeholder,

                      decoder_inputs_placeholder,

                      initial_s,

                      initial_c],

              outputs=outputs)
# compile the model

model.compile(optimizer='adam',loss='categorical_crossentropy',

              metrics=['accuracy'])
z = np.zeros((len(encoder_inputs), LATENT_DIM_DECODER)) # initial [s, c]

r = model.fit(

  [encoder_inputs, decoder_inputs, z, z], decoder_targets_one_hot,

  batch_size=BATCH_SIZE,

  epochs=15,

  validation_split=0.2

)
## Make predictions

encoder_model = Model(encoder_inputs_placeholder, encoder_outputs)



# T=1 decoder model

encoder_outputs_as_input = Input(shape=(max_len_input, LATENT_DIM*2))

decoder_inputs_single = Input(shape=(1,))

decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
context = one_step_attention(encoder_outputs_as_input, initial_s)

decoder_lstm_input = context_last_word_concat_layer([context,decoder_inputs_single_x])
o,s,c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s,initial_c])

decoder_outputs = decoder_dense(o)
decoder_model = Model(inputs=[decoder_inputs_single,

                              encoder_outputs_as_input,

                              initial_s,

                              initial_c],

                      outputs=[decoder_outputs, s, c])
# map indices back to real words

idx2word_eng = {v:k for k,v in word2idx_inputs.items()}

idx2word_trans = {v:k for k,v in word2idx_outputs.items()}
def decode_sequence(input_seq):

  enc_out = encoder_model.predict(input_seq)

  

  target_seq = np.zeros((1,1))

  target_seq[0,0] = word2idx_outputs['<sos>']

  

  eos = word2idx_outputs['<eos>']

  

  s = np.zeros((1, LATENT_DIM_DECODER))

  c = np.zeros((1, LATENT_DIM_DECODER))

  

  # creating the translation

  output_sentence = []

  for _ in range(max_len_target):

    o,s,c = decoder_model.predict([target_seq, enc_out, s, c])

    

    idx = np.argmax(o.flatten())

    

    if eos==idx:

      break

      

    word = ''

    

    if idx>0:

      word = idx2word_trans[idx]

      output_sentence.append(word)

    

    target_seq[0,0] = idx

  

  return ' '.join(output_sentence)
# i = np.random.choice(len(input_texts))

# input_seq = encoder_inputs[i:i+1]

# translation = decode_sequence(input_seq)

# print('Input Sentence:',input_texts[i])

# print('Predicted Translation:',translation)