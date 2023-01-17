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
import numpy as np

import tensorflow as tf

from tensorflow import keras
df = pd.read_json('../input/arxiv-papers-2010-2020/arXiv_title_abstract_20200809_2011_2020.json')

df.head()
val_df = df.sample(frac=0.1, random_state=1007)

train_df = df.drop(val_df.index)

test_df = train_df.sample(frac=0.1, random_state=1007)

train_df.drop(test_df.index, inplace=True)
# your code here 
import gc

del df

gc.collect()
train_samples = train_df.sample(frac=0.2, random_state=1007)


train_samples.head()
import string

from string import digits

import re

exclude = set(string.punctuation) # Set of all special characters



# Lowercase all characters

train_samples['abstract']=train_samples['abstract'].apply(lambda x: x.lower())

train_samples['title']=train_samples['title'].apply(lambda x: x.lower())



val_df['abstract']=val_df['abstract'].apply(lambda x: x.lower())

val_df['title']=val_df['title'].apply(lambda x: x.lower())





# Remove quotes

#train_samples['abstract']=train_samples['abstract'].apply(lambda x: re.sub(r'[a-z]*\n[a-z]', '@#@', x))

#train_samples['title']=train_samples['title'].apply(lambda x: re.sub(r'[\n\r]+', '@#@', x))





# Remove all the special characters

train_samples['abstract']=train_samples['abstract'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

train_samples['title']=train_samples['title'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))



val_df['abstract']=val_df['abstract'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

val_df['title']=val_df['title'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))



# Remove all numbers from text

remove_digits = str.maketrans('', '', digits)

train_samples['abstract']=train_samples['abstract'].apply(lambda x: x.translate(remove_digits))

train_samples['title'] = train_samples['title'].apply(lambda x: x.translate(remove_digits))

val_df['abstract']=val_df['abstract'].apply(lambda x: x.translate(remove_digits))

val_df['title'] = val_df['title'].apply(lambda x: x.translate(remove_digits))



# Remove extra spaces

train_samples['abstract']=train_samples['abstract'].apply(lambda x: x.strip())

train_samples['title']=train_samples['title'].apply(lambda x: x.strip())



train_samples['abstract']=train_samples['abstract'].apply(lambda x: re.sub(" +", " ", x))

train_samples['title']=train_samples['title'].apply(lambda x: re.sub(" +", " ", x))



val_df['abstract']=val_df['abstract'].apply(lambda x: x.strip())

val_df['title']=val_df['title'].apply(lambda x: x.strip())



val_df['abstract']=val_df['abstract'].apply(lambda x: re.sub(" +", " ", x))

val_df['title']=val_df['title'].apply(lambda x: re.sub(" +", " ", x))



# Remove '\n'

train_samples['abstract']=train_samples['abstract'].apply(lambda x: x.replace('\n',' '))

train_samples['title']=train_samples['title'].apply(lambda x: x.replace('\n',' '))



val_df['abstract']=val_df['abstract'].apply(lambda x: x.replace('\n',' '))

val_df['title']=val_df['title'].apply(lambda x: x.replace('\n',' '))





#train_samples['abstract'][61322]
input_texts=train_samples['abstract']

target_texts='\t'+train_samples['title']+'\n'
target_texts.head()

input_characters=set()

for txt in input_texts:

    for char in txt:

        if char not in input_characters:

            input_characters.add(char)



print(input_characters)
print('length of input list characters','   ',len(input_characters))
input_characters = sorted(list(input_characters))

print(input_characters)
target_characters=set()

for txt in target_texts:

    for char in txt:

        if char not in target_characters:

            target_characters.add(char)



print(target_characters)
print('length of target list characters','   ',len(target_characters))
target_characters = sorted(list(target_characters))

print(target_characters)
#input_characters = sorted(list(input_characters))

#target_characters = sorted(list(target_characters))

num_encoder_tokens = len(input_characters)

num_decoder_tokens = len(target_characters)

max_encoder_seq_length = max([len(txt) for txt in input_texts])

max_decoder_seq_length =max([len(txt) for txt in target_texts])



print('Number of samples:', len(input_texts))

print('Number of unique input tokens:', num_encoder_tokens)

print('Number of unique output tokens:', num_decoder_tokens)

print('Max sequence length for inputs:', max_encoder_seq_length)

print('Max sequence length for outputs:', max_decoder_seq_length)
input_token_index = dict(

  [(char, i) for i, char in enumerate(input_characters)])

target_token_index = dict(

  [(char, i) for i, char in enumerate(target_characters)])

import numpy as np



encoder_input_data = np.zeros(

  (len(input_texts), max_encoder_seq_length, num_encoder_tokens),

  dtype='float32')

decoder_input_data = np.zeros(

  (len(input_texts), max_decoder_seq_length, num_decoder_tokens),

  dtype='float32')

decoder_target_data = np.zeros(

  (len(input_texts), max_decoder_seq_length, num_decoder_tokens),

  dtype='float32')

encoder_input_data.shape
decoder_input_data.shape
decoder_target_data.shape
for i, (input_text, target_text) in enumerate(zip(input_texts,  target_texts)):

    for t, char in enumerate(input_text):

        encoder_input_data[i, t, input_token_index[char]] = 1.

    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.

    for t, char in enumerate(target_text):

        # decoder_target_data is ahead of decoder_input_data by one timestep

        decoder_input_data[i, t , target_token_index[char]] = 1.

        if t > 0:

          # decoder_target_data will be ahead by one timestep

          # and will not include the start character.

            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.

    decoder_target_data[i, t:, target_token_index[' ']] = 1.

    
batch_size = 64  # Batch size for training.

epochs = 50  # Number of epochs to train for.

latent_dim = 256  # Latent dimensionality of the encoding space.
from keras.layers import Input, LSTM, Dense

from keras.models import Model

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



decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation="softmax")

decoder_outputs = decoder_dense(decoder_outputs)









# Define the model that will turn

# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()
model.compile(

    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]

)

model.fit(

    [encoder_input_data, decoder_input_data],

    decoder_target_data,

    batch_size=batch_size,

    epochs=epochs,

    validation_split=0.2,

)



# Save model

model.save("s2s.h5")
from keras.models import Model,load_model



model = load_model("s2s.h5")
from keras.models import Model

from keras.layers import Input, LSTM, Dense
# Here's the drill:

# 1) encode input and retrieve initial decoder state

# 2) run one step of decoder with this initial state

# and a "start of sequence" token as target.

# Output will be the next target token

# 3) Repeat with the current target token and current states

# Define sampling models

encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(latent_dim,))

decoder_state_input_c = Input(shape=(latent_dim,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(

    decoder_inputs, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(

    [decoder_inputs] + decoder_states_inputs,

    [decoder_outputs] + decoder_states)
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

    target_seq[0, 0] = 1.



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

#for seq_index in range(0,1):

    ## Take one sequence (part of the training set)

    ## for trying out decoding.

#    input_seq = encoder_input_data[seq_index: seq_index + 1]

#    decoded_sentence = decode_sequence(input_seq)

#    print('-')

#    print('Input sentence:', input_texts[seq_index],'\n')

#    print('Decoded sentence:', decoded_sentence)
#from nltk.translate.bleu_score import sentence_bleu

    

#score = sentence_bleu(input_texts[seq_index], target_texts[seq_index])

#print('from original sentence',score)

    

#score = sentence_bleu(input_texts[seq_index], decoded_sentence)

#print('from converted sentence',score)
# your code here
test_df
# Lowercase all characters



test_df['abstract']=test_df['abstract'].apply(lambda x: x.lower())

test_df['title']=test_df['title'].apply(lambda x: x.lower())





# Remove quotes

#train_samples['abstract']=train_samples['abstract'].apply(lambda x: re.sub(r'[a-z]*\n[a-z]', '@#@', x))

#train_samples['title']=train_samples['title'].apply(lambda x: re.sub(r'[\n\r]+', '@#@', x))





# Remove all the special characters



test_df['abstract']=test_df['abstract'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

test_df['title']=test_df['title'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))



# Remove all numbers from text

remove_digits = str.maketrans('', '', digits)



test_df['abstract']=test_df['abstract'].apply(lambda x: x.translate(remove_digits))

test_df['title'] = test_df['title'].apply(lambda x: x.translate(remove_digits))



# Remove extra spaces





test_df['abstract']=test_df['abstract'].apply(lambda x: x.strip())

test_df['title']=test_df['title'].apply(lambda x: x.strip())



test_df['abstract']=test_df['abstract'].apply(lambda x: re.sub(" +", " ", x))

test_df['title']=test_df['title'].apply(lambda x: re.sub(" +", " ", x))



# Remove '\n'



test_df['abstract']=test_df['abstract'].apply(lambda x: x.replace('\n',' '))

test_df['title']=test_df['title'].apply(lambda x: x.replace('\n',' '))



###### test



from nltk.translate.bleu_score import sentence_bleu







for col,raw in enumerate(test_df.index[0:5]):

    input_sentence = test_df['abstract'][raw]

    actual_tile=test_df['title'][raw]



    test_sentence_tokenized = np.zeros(

      (1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')



    for t, char in enumerate(input_sentence):

        test_sentence_tokenized[0, t, input_token_index[char]] = 1.



    

    

    

    

    print("........Abstract.........\n",input_sentence,'\n')

    print(".........Actual Title.........\n",actual_tile,'\n')

    print('.........Predicted Title..........\n',decode_sequence(test_sentence_tokenized))

    

    score = sentence_bleu(input_sentence[col], decode_sequence(test_sentence_tokenized))

    print('.......BLEU ===> fom converted sentence.................',score)



    print ('--------------------------------------')