%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

# File loading

df  = pd.read_csv('../input/unigram_freq.csv')

print(df.shape)

df.dropna(axis=0,how='any')

print(df.shape)

lines = [x for x in df['word'] if type(x) == type('a') ]

print("Line Count:",len(lines))

print(lines[:4])
# Preprocessing

import re

def process(sent):

    sent=sent.lower()

    sent=re.sub(r'[^0-9a-zA-Z ]','',sent)

    sent=sent.replace('\n','')

    return sent    
lines =[process(x) for x in lines]

temp = []

for line in lines:

    temp+= [ x for x in line.split() ]

lines = list(set(temp))

print("\n".join(lines[:4]))

print("Number of items:",len(lines))
# CHAR INDEXING

char_set = list(" abcdefghijklmnopqrstuvwxyz0123456789")

char2int = { char_set[x]:x for x in range(len(char_set)) }

int2char = { char2int[x]:x for x in char_set }

print(char2int)

print(int2char)
count = len(char_set)

codes = ["\t","\n",'#']

for i in range(len(codes)):

    code = codes[i]

    char2int[code]=count

    int2char[count]=code

    count+=1

print(char2int)

print(int2char)
import random

#thresh - 0 to 1

def gen_gibberish(line,thresh=0.2):

    times = int(random.randrange(1,len(line)) * thresh)

    '''

    Types of replacement:

        1.Delete random character.

        2.Add random character.

        3.Replace a character.

        4.Combination?

    '''

    while times!=0:

        # try to gen noise length times...

        times-=1

        val = random.randrange(0,10)

        if val <= 5:

            #get random index

            val = random.randrange(0,10)

            index = random.randrange(2,len(line))

            if val <= 3 :

                #delete character

                line = line[:index]+line[index+1:]

            else:

                #add character

                insert_index = random.randrange(0,len(char_set))

                line = line[:index] + char_set[insert_index] + line[index:]

        else:

            index = random.randrange(0,len(char_set))

            replace_index = random.randrange(2,len(line))

            line = line[:replace_index] + char_set[index] + line[replace_index+1:]

    return line



sample = lines[5]

gib = gen_gibberish(sample)

print("Original:",sample)

print("Gibberish:",gib)

        

    
# create dataset

input_texts = []

target_texts = []

REPEAT_FACTOR = 1

SKIP = int(len(lines)*0.65)



for line in lines[SKIP:]:

    if len(line)>10:

        output_text = '\t' + line + '\n'

        for _ in range(REPEAT_FACTOR):

            input_text = gen_gibberish(line)

            input_texts.append(input_text)

            target_texts.append(output_text)

print("LEN OF SAMPLES:",len(input_texts))
max_enc_len = max([len(x) for x in input_texts])

max_dec_len = max([len(x) for x in target_texts])

print("Max Enc Len:",max_enc_len)

print("Max Dec Len:",max_dec_len)
num_samples = len(input_texts)

encoder_input_data = np.zeros( (num_samples , max_enc_len , len(char_set)),dtype='float32' )

decoder_input_data = np.zeros( (num_samples , max_dec_len , len(char_set)+2),dtype='float32' )

decoder_target_data = np.zeros( (num_samples , max_dec_len , len(char_set)+2),dtype='float32' )

print("CREATED ZERO VECTORS")
#filling in the enc,dec datas

for i,(input_text,target_text) in enumerate(zip(input_texts,target_texts)):

    for t,char in enumerate(input_text):

        encoder_input_data[ i , t , char2int[char] ] = 1

    for t,char in enumerate(target_text):

        decoder_input_data[ i, t , char2int[char] ] = 1

        if t > 0 :

            decoder_target_data[ i , t-1 , char2int[char] ] = 1

print("COMPLETED...")    
from keras.models import Model

from keras.layers import Input,LSTM,Dense
batch_size = 128

epochs = 1000

latent_dim = 256



num_enc_tokens = len(char_set)

num_dec_tokens = len(char_set) + 2 # includes \n \t

encoder_inputs = Input(shape=(None,num_enc_tokens))

encoder = LSTM(latent_dim,return_state=True)

encoder_outputs , state_h, state_c = encoder(encoder_inputs)

encoder_states = [state_h,state_c]

decoder_inputs = Input(shape=(None,num_dec_tokens))

decoder_lstm = LSTM(latent_dim,return_sequences=True,return_state=True)

decoder_ouputs,_,_ = decoder_lstm(decoder_inputs,initial_state = encoder_states)



decoder_dense = Dense(num_dec_tokens, activation='softmax')

decoder_ouputs = decoder_dense(decoder_ouputs)



model = Model([encoder_inputs,decoder_inputs],decoder_ouputs)

model.compile(optimizer='rmsprop',loss='categorical_crossentropy')

model.summary()
h=model.fit([encoder_input_data,decoder_input_data],decoder_target_data

         ,epochs = epochs,

          batch_size = batch_size,

          validation_split = 0.2

         )

model.save('s2s.h5')

plt.plot(h.history['loss'])

plt.title('Model Loss')

plt.show()
encoder_model = Model(encoder_inputs,encoder_states)



decoder_state_input_h = Input(shape=(latent_dim,))

decoder_state_input_c = Input(shape=(latent_dim,))

decoder_states_inputs = [decoder_state_input_h,decoder_state_input_c]

decoder_outputs,state_h,state_c = decoder_lstm(

        decoder_inputs,initial_state = decoder_states_inputs

)

decoder_states = [state_h,state_c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(

    [decoder_inputs] + decoder_states_inputs,

    [decoder_outputs] + decoder_states

)

encoder_model.save('encoder.h5')

decoder_model.save('decoder.h5')



def decode_sequence(input_seq):

    # Encode the input as state vectors.

    states_value = encoder_model.predict(input_seq)



    # Generate empty target sequence of length 1.

    target_seq = np.zeros((1, 1, num_dec_tokens))

    # Populate the first character of target sequence with the start character.

    target_seq[0, 0, char2int['\t']] = 1.



    # Sampling loop for a batch of sequences

    # (to simplify, here we assume a batch of size 1).

    stop_condition = False

    decoded_sentence = ''

    while not stop_condition:

        output_tokens, h, c = decoder_model.predict(

            [target_seq] + states_value)



        # Sample a token

        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        sampled_char = int2char[sampled_token_index]

        decoded_sentence += sampled_char



        # Exit condition: either hit max length

        # or find stop character.

        if (sampled_char == '\n' or

           len(decoded_sentence) > max_dec_len):

            stop_condition = True



        # Update the target sequence (of length 1).

        target_seq = np.zeros((1, 1, num_dec_tokens))

        target_seq[0, 0, sampled_token_index] = 1.



        # Update states

        states_value = [h, c]



    return decoded_sentence



for seq_index in range(10):

    input_seq = encoder_input_data[seq_index: seq_index + 1]

    decoded_sentence = decode_sequence(input_seq)

    print('-')

    print('Wrong sentence:', input_texts[seq_index])

    print('Corrected sentence:', decoded_sentence)

    print('Ground Truth:',target_texts[seq_index])