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
import pandas as pd
import numpy as np
import string
from string import digits
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import re
from sklearn.model_selection import train_test_split

from keras.layers import Input, LSTM, Embedding, Dense, Bidirectional, Concatenate, Dot, Activation, TimeDistributed
from keras.models import Model
from keras.utils import plot_model

batch_size=64
epochs=100
latent_dim=256
num_samples=10000

data_path='../input/fra-eng/fra.txt'
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

# Vectorize the data.
input_texts = []
target_bef = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_bef.append(target_text)
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
target_texts
#char level
line = pd.DataFrame({'input':input_texts, 'target':target_texts})
line
lines = pd.DataFrame({'input':input_texts, 'target':target_bef})
lines
def cleanup(lines):
    
      # Since we work on word level, if we normalize the text to lower case, this will reduce the vocabulary. It's easy to recover the case later. 
    lines.input=lines.input.apply(lambda x: x.lower())
    lines.target=lines.target.apply(lambda x: x.lower())

    # To help the model capture the word separations, mark the comma with special token:
    lines.input=lines.input.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))
    lines.target=lines.target.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))

    # Clean up punctuations and digits. Such special chars are common to both domains, and can just be copied with no error.
    exclude = set(string.punctuation)
    lines.input=lines.input.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    lines.target=lines.target.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

    remove_digits = str.maketrans('', '', digits)
    lines.input=lines.input.apply(lambda x: x.translate(remove_digits))
    lines.target=lines.target.apply(lambda x: x.translate(remove_digits))

      #return lines

st_tok = 'START_'
end_tok = '_END'
def data_prep(lines):
    cleanup(lines)
    lines.target = lines.target.apply(lambda x : st_tok + ' ' + x + ' ' + end_tok)
data_prep(lines)
lines.head()
#word_leve
def tok_split_word2word(data):
    return data.split()
tok_split_fn = tok_split_word2word
def data_stats(lines, input_tok_split_fn, target_tok_split_fn):
    input_tokens=set()
    for line in lines.input:
        for tok in input_tok_split_fn(line):
            if tok not in input_tokens:
                input_tokens.add(tok)
      
    target_tokens=set()
    for line in lines.target:
        for tok in target_tok_split_fn(line):
            if tok not in target_tokens:
                target_tokens.add(tok)
    input_tokens = sorted(list(input_tokens))
    target_tokens = sorted(list(target_tokens))

    num_encoder_tokens = len(input_tokens)
    num_decoder_tokens = len(target_tokens)
    max_encoder_seq_length = np.max([len(input_tok_split_fn(l)) for l in lines.input])
    max_decoder_seq_length = np.max([len(target_tok_split_fn(l)) for l in lines.target])

    return input_tokens, target_tokens, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length


input_tokens, target_tokens, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length = data_stats(lines, input_tok_split_fn=tok_split_fn, target_tok_split_fn=tok_split_fn)
print('Number of samples:', len(lines))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
input_tokens
target_tokens
pad_tok = 'PAD'
sep_tok = ' '
special_tokens = [pad_tok, sep_tok, st_tok, end_tok] 
num_encoder_tokens += len(special_tokens)
num_decoder_tokens += len(special_tokens)
#adding the nwew value of thr
def vocab(input_tokens, target_tokens):
    
    input_token_index = {}
    target_token_index = {}
    for i,tok in enumerate(special_tokens):

        input_token_index[tok] = i
        target_token_index[tok] = i 

    offset = len(special_tokens)
    for i, tok in enumerate(input_tokens):
        input_token_index[tok] = i+offset

    for i, tok in enumerate(target_tokens):
        target_token_index[tok] = i+offset
    # Reverse-lookup token index to decode sequences back to something readable.
    reverse_input_tok_index = dict(
        (i, tok) for tok, i in input_token_index.items())
    reverse_target_tok_index = dict(
        (i, tok) for tok, i in target_token_index.items())
    return input_token_index, target_token_index, reverse_input_tok_index, reverse_target_tok_index
input_token_index, target_token_index, reverse_input_tok_index, reverse_target_tok_index = vocab(input_tokens, target_tokens)
max_encoder_seq_length = 16
max_decoder_seq_length = 16
def init_model_inputs(lines, max_encoder_seq_length, max_decoder_seq_length, num_decoder_tokens):
    encoder_input_data = np.zeros(
        (len(lines.input), max_encoder_seq_length),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(lines.target), max_decoder_seq_length),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(lines.target), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
  
    return encoder_input_data, decoder_input_data, decoder_target_data
def vectorize(lines, max_encoder_seq_length, max_decoder_seq_length, num_decoder_tokens, input_tok_split_fn, target_tok_split_fn):
    encoder_input_data, decoder_input_data, decoder_target_data = init_model_inputs(lines, max_encoder_seq_length, max_decoder_seq_length, num_decoder_tokens)
    for i, (input_text, target_text) in enumerate(zip(lines.input, lines.target)):
        for t, tok in enumerate(input_tok_split_fn(input_text)):
            encoder_input_data[i, t] = input_token_index[tok]
        encoder_input_data[i, t+1:] = input_token_index[pad_tok]
        for t, tok in enumerate(target_tok_split_fn(target_text)):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t] = target_token_index[tok]         
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[tok]] = 1.
        decoder_input_data[i, t+1:] = target_token_index[pad_tok] 
        decoder_target_data[i, t:, target_token_index[pad_tok]] = 1.          
              
    return encoder_input_data, decoder_input_data, decoder_target_data              
encoder_input_data, decoder_input_data, decoder_target_data  = vectorize(lines, max_encoder_seq_length, max_decoder_seq_length, num_decoder_tokens, input_tok_split_fn=tok_split_fn, target_tok_split_fn=tok_split_fn)
def seq2seq(num_decoder_tokens, num_encoder_tokens, emb_sz, lstm_sz):
    encoder_inputs = Input(shape=(None,))
    en_x=  Embedding(num_encoder_tokens, emb_sz)(encoder_inputs)
    encoder = LSTM(lstm_sz, return_state=True)
    encoder_outputs, state_h, state_c = encoder(en_x)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
  
    # Encoder model
    encoder_model = Model(encoder_inputs, encoder_states)
  
   
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
 
    dex=  Embedding(num_decoder_tokens, emb_sz)

    final_dex= dex(decoder_inputs)
  
  
    decoder_lstm = LSTM(lstm_sz, return_sequences=True, return_state=True)

    decoder_outputs, _, _ = decoder_lstm(final_dex,
                                      initial_state=encoder_states)
  
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')

    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])


  
    # Decoder model: Re-build based on explicit state inputs. Needed for step-by-step inference:
    decoder_state_input_h = Input(shape=(lstm_sz,))
    decoder_state_input_c = Input(shape=(lstm_sz,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)
    decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)  

    return model, encoder_model, decoder_model

emb_sz=50
model, encoder_model, decoder_model = seq2seq(num_decoder_tokens, num_encoder_tokens, emb_sz, emb_sz)
print(model.summary())
plot_model(model, show_shapes=True, show_layer_names=True)
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=64,
          epochs=30,
          validation_split=0.05)
def seq2seq(num_decoder_tokens, num_encoder_tokens, emb_sz, lstm_sz):
  encoder_inputs = Input(shape=(None,))
  en_x=  Embedding(num_encoder_tokens, emb_sz, mask_zero=True)(encoder_inputs)
  encoder = LSTM(lstm_sz, return_state=True)
  encoder_outputs, state_h, state_c = encoder(en_x)
  # We discard `encoder_outputs` and only keep the states.
  encoder_states = [state_h, state_c]
  
  # Encoder model
  encoder_model = Model(encoder_inputs, encoder_states)
  
  
  # Set up the decoder, using `encoder_states` as initial state.
  decoder_inputs = Input(shape=(None,))

  dex=  Embedding(num_decoder_tokens, emb_sz, mask_zero=True)

  final_dex= dex(decoder_inputs)


  decoder_lstm = LSTM(lstm_sz, return_sequences=True, return_state=True)

  decoder_outputs, _, _ = decoder_lstm(final_dex,
                                      initial_state=encoder_states)

  decoder_dense = Dense(num_decoder_tokens, activation='softmax')

  decoder_outputs = decoder_dense(decoder_outputs)

  model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])


  
  # Decoder model: Re-build based on explicit state inputs. Needed for step-by-step inference:
  decoder_state_input_h = Input(shape=(lstm_sz,))
  decoder_state_input_c = Input(shape=(lstm_sz,))
  decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

  decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex, initial_state=decoder_states_inputs)
  decoder_states2 = [state_h2, state_c2]
  decoder_outputs2 = decoder_dense(decoder_outputs2)
  decoder_model = Model(
  [decoder_inputs] + decoder_states_inputs,
  [decoder_outputs2] + decoder_states2)  

  return model, encoder_model, decoder_model
emb_sz = 256
model, encoder_model, decoder_model = seq2seq(num_decoder_tokens, num_encoder_tokens, emb_sz, emb_sz)
print(model.summary())
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=64,
          epochs=10,
          validation_split=0.2)
#using the last decoder which has been traind on the GT 
def decode_sequence(input_seq, sep=' '):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index[st_tok]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_tok = reverse_target_tok_index[sampled_token_index]
        decoded_sentence += sep + sampled_tok

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_tok == end_tok or
           len(decoded_sentence) > 52):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence
for seq_index in range(100): #[14077,20122,40035,40064, 40056, 40068, 40090, 40095, 40100, 40119, 40131, 40136, 40150, 40153]:
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', lines.input[seq_index: seq_index + 1])
    print('Decoded sentence:', decoded_sentence)
def seq2seq_attention(num_encoder_tokens, num_decoder_tokens, emb_sz, latent_dim):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None,), dtype='float32')
    encoder_inputs_ = Embedding(num_encoder_tokens, emb_sz, mask_zero=True)(encoder_inputs)    
    
    encoder = Bidirectional(LSTM(latent_dim, return_state=True, return_sequences=True)) # Bi LSTM
    encoder_outputs, state_f_h, state_f_c, state_b_h, state_b_c = encoder(encoder_inputs_)# Bi LSTM
    state_h = Concatenate()([state_f_h, state_b_h])# Bi LSTM
    state_c = Concatenate()([state_f_c, state_b_c])# Bi LSTM

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]# Bi GRU, LSTM, BHi LSTM
    print(encoder_states)
    
    decoder_inputs = Input(shape=(None,))
    decoder_inputs_ = Embedding(num_decoder_tokens, emb_sz, mask_zero=True)(decoder_inputs)    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim*2, return_sequences=True, return_state=True)# Bi LSTM
    
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs_, initial_state=encoder_states)

    # Equation (7) with 'dot' score from Section 3.1 in the paper.
    # Note that we reuse Softmax-activation layer instead of writing tensor calculation
    print(decoder_outputs)
    print(encoder_outputs)
    att_dot = Dot(axes=[2, 2])
    attention = att_dot([decoder_outputs, encoder_outputs])
    att_activation = Activation('softmax', name='attention')
    attention = att_activation(attention)
    print('attention', attention)
    context_dot = Dot(axes=[2,1])
    context = context_dot([attention, encoder_outputs])
    att_context_concat = Concatenate()
    decoder_combined_context = att_context_concat([context, decoder_outputs])

    # Has another weight + tanh layer as described in equation (5) of the paper

    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    #decoder_outputs = decoder_dense(decoder_outputs)
    decoder_outputs = decoder_dense(decoder_combined_context)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    print('encoder-decoder  model:')
    print(model.summary()) 
    
    print(encoder_inputs)
    print(encoder_outputs)
    print(encoder_states)
    encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs] + encoder_states)

    decoder_encoder_inputs = Input(shape=(None, latent_dim*2,))
    decoder_state_input_h = Input(shape=(latent_dim*2,))# Bi LSTM
    decoder_state_input_c = Input(shape=(latent_dim*2,)) # Bi LSTM
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs_, initial_state=decoder_states_inputs)

    
    decoder_states = [state_h, state_c]
    
    # Equation (7) with 'dot' score from Section 3.1 in the paper.
    # Note that we reuse Softmax-activation layer instead of writing tensor calculation
    
    attention = att_dot([decoder_outputs, decoder_encoder_inputs])
    
    attention = att_activation(attention)
    
    context = context_dot([attention, decoder_encoder_inputs])
    
    
    
    decoder_combined_context = att_context_concat([context, decoder_outputs])
    
    # Has another weight + tanh layer as described in equation (5) of the paper
    
    decoder_outputs = decoder_dense(decoder_combined_context)
    
    decoder_model = Model(
        [decoder_inputs, decoder_encoder_inputs] + decoder_states_inputs,
        [decoder_outputs, attention] + decoder_states)
    
    return model, encoder_model, decoder_model
model, encoder_model, decoder_model = seq2seq_attention(num_encoder_tokens, num_decoder_tokens, emb_sz=emb_sz, latent_dim=emb_sz)
print(model.summary())
plot_model(model, show_shapes=True, show_layer_names=True)

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=128,
          epochs=20,
          validation_split=0.05)
def decode_sequence_attention(input_seq, sep=' '):
    # Encode the input as state vectors.
    encoder_outputs, h, c = encoder_model.predict(input_seq)
    states_value = [h,c]
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index[st_tok]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    attention_density = []
    while not stop_condition:
        output_tokens, attention, h, c  = decoder_model.predict(
            [target_seq, encoder_outputs] + states_value)
        attention_density.append(attention[0][0])# attention is max_sent_len x 1 since we have num_time_steps = 1 for the output
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_tok = reverse_target_tok_index[sampled_token_index]
        decoded_sentence += sep + sampled_tok

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_tok == end_tok or
           len(decoded_sentence) > 52):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]
    attention_density = np.array(attention_density)
    return decoded_sentence, attention_density
word_decoded_sents = []
for seq_index in range(100): #[14077,20122,40035,40064, 40056, 40068, 40090, 40095, 40100, 40119, 40131, 40136, 40150, 40153]:
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence, attention = decode_sequence_attention(input_seq)
    print('-')
    print('Input sentence:', lines.input[seq_index: seq_index + 1])
    print('Decoded sentence:', decoded_sentence)
    word_decoded_sents.append(decoded_sentence)
lines = pd.DataFrame({'input':input_texts, 'target':target_bef})
num_samples = 10000
lines = lines[:num_samples]
st_tok = '\t'
end_tok = '\n'
def data_prep(lines):
  #cleanup(lines)
  lines.target = lines.target.apply(lambda x : st_tok  + x  + end_tok)
  
data_prep(lines)

lines
#using chracter level
emb_sz = 256
def tok_split_char2char(data):
    return data
  
tok_split_fn = tok_split_char2char
input_tokens, target_tokens, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length = data_stats(lines, input_tok_split_fn=tok_split_fn, target_tok_split_fn=tok_split_fn)
print('Number of samples:', len(lines))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
pad_tok = 'PAD'
sep_tok = ' '
special_tokens = [pad_tok, sep_tok, st_tok, end_tok] 
num_encoder_tokens += len(special_tokens)
num_decoder_tokens += len(special_tokens)

input_token_index, target_token_index, reverse_input_tok_index, reverse_target_tok_index = vocab(input_tokens, target_tokens)
encoder_input_data, decoder_input_data, decoder_target_data  = vectorize(lines, max_encoder_seq_length, max_decoder_seq_length, num_decoder_tokens, input_tok_split_fn=tok_split_fn, target_tok_split_fn=tok_split_fn)
emb_sz = 256
model, encoder_model, decoder_model = seq2seq(num_decoder_tokens, num_encoder_tokens, emb_sz, emb_sz)
print(model.summary())
plot_model(model, show_shapes=True, show_layer_names=True)
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=64,
          epochs=10,
          validation_split=0.2)
for seq_index in range(100): #[14077,20122,40035,40064, 40056, 40068, 40090, 40095, 40100, 40119, 40131, 40136, 40150, 40153]:
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq, sep='')
    print('-')
    print('Input sentence:', lines.input[seq_index: seq_index + 1])
    print('Decoded sentence:', decoded_sentence)
model, encoder_model, decoder_model = seq2seq_attention(num_encoder_tokens, num_decoder_tokens, emb_sz=emb_sz, latent_dim=emb_sz)
print(model.summary())
plot_model(model)
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=128,
          epochs=20,
          validation_split=0.05)
char_decoded_sents = []
target_sents = []
for seq_index in range(100): #[14077,20122,40035,40064, 40056, 40068, 40090, 40095, 40100, 40119, 40131, 40136, 40150, 40153]:
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence, attention = decode_sequence_attention(input_seq, sep='')
    print('-')
    print('Input sentence:', lines.input[seq_index: seq_index + 1])
    print('GT sentence:', lines.target[seq_index: seq_index + 1][1:-1])
    print('Decoded sentence:', decoded_sentence)
    char_decoded_sents.append(decoded_sentence)
    target_sents.append(np.array(lines.target[seq_index: seq_index + 1]))