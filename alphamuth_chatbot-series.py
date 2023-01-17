# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import tensorflow as tf

import re

import numpy as np

tf.__version__
southpark=pd.read_csv('../input/southparklines/All-seasons.csv')
southpark.head()
def clean_text(text):

    text = text.lower()

    

    text = re.sub(r"\n", "",  text)

    text = re.sub(r"[-()]", "", text)

    text = re.sub(r"\.", " .", text)

    text = re.sub(r"\!", " !", text)

    text = re.sub(r"\?", " ?", text)

    text = re.sub(r"\,", " ,", text)

    text = re.sub(r"i'm", "i am", text)

    text = re.sub(r"he's", "he is", text)

    text = re.sub(r"she's", "she is", text)

    text = re.sub(r"it's", "it is", text)

    text = re.sub(r"that's", "that is", text)

    text = re.sub(r"what's", "what is", text)

    text = re.sub(r"\'ll", " will", text)

    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"won't", "will not", text)

    text = re.sub(r"can't", "cannot", text)

    text = re.sub(r"n't", " not", text)

    text = re.sub(r"n'", "ng", text)

    text = re.sub(r"ohh", "oh", text)

    text = re.sub(r"ohhh", "oh", text)

    text = re.sub(r"ohhhh", "oh", text)

    text = re.sub(r"ohhhhh", "oh", text)

    text = re.sub(r"ohhhhhh", "oh", text)

    text = re.sub(r"ahh", "ah", text)

    

    return text
text=[]

for line in southpark.Line:

    text.append(clean_text(line))

#counting length of each sentence by splitting a sentence into words

length=[]



for line in text:

    #print(line.split())

    length.append(len(line.split()))

lengths = pd.DataFrame(length, columns=['counts'])
lengths.describe()
print(np.percentile(lengths, 80))

print(np.percentile(lengths, 85))

print(np.percentile(lengths, 90))

print(np.percentile(lengths, 95))



print(np.percentile(lengths, 99))

max_line_len=30

short_text=[]

for line in text:

    if(len(line.split() )<= max_line_len):

        short_text.append(line)

short_text[:5]

vocab={}

for line in short_text:

    for word in line.split():

        if word not in vocab:

            vocab[word]=1

        else :

            vocab[word]+=1
print("Size of vocab is",len(vocab))
#limit occurance of word, that is used more than 3 times

threshold = 3

count=0

for k,v in vocab.items():

    

    if v>=threshold:

        count+=1

    else: 

        #print(v)

        pass

#count

print("Size of total vocab:", len(vocab))

print("Size of vocab we will use:", count)
source_vocab_to_int = {}

word_num=0

for k,v in vocab.items():

    if v >= threshold:

        source_vocab_to_int[k]=word_num

        word_num+=1

len(source_vocab_to_int)

target_vocab_to_int={}

word_num=0

for k,v in vocab.items():

    if v>= threshold:

        target_vocab_to_int[k]=word_num

        word_num+=1

len(target_vocab_to_int)
# adding essential token to the vocab (dictionary)

tokens = ['<PAD>','<EOS>','<UNK>','<GO>']

for token in tokens:

    source_vocab_to_int[token]=len(source_vocab_to_int)+1

for token in tokens:

    target_vocab_to_int[token]=len(target_vocab_to_int)+1
# int to vocab mapping

source_int_to_vocab={v_i:v for v,v_i in source_vocab_to_int.items()}

target_int_to_vocab={v_i:v for v,v_i in target_vocab_to_int.items()}

# Check the length of the dictionaries.

print(len(source_vocab_to_int))

print(len(source_int_to_vocab))

print(len(target_vocab_to_int))

print(len(target_int_to_vocab))
# creating source and  target text

source_text = short_text[:-1]

target_text = short_text[1:]



for i in range(len(target_text)):

    target_text[i] += ' <EOS>'
print(len(source_text))

print(len(target_text))
source_text[:3]


import sys

import os

import pandas as pd

import numpy as np

import re

import nltk

from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Bidirectional,GRU,CuDNNLSTM,CuDNNGRU

from keras.models import Model, load_model



num_samples=3000

short_source=source_text[:3000]

short_target=target_text[:3000]

short_source_token = [nltk.word_tokenize(sent) for sent in short_source]

short_target_token = [nltk.word_tokenize(sent) for sent in short_target]
print(short_source_token[:5])

print(short_target_token[:5])
data_size = len(short_source_token)



# We will use the first 0-80th %-tile (80%) of data for the training

training_input  = short_source_token[:round(data_size*(80/100))]

training_input  = [tr_input[::-1] for tr_input in training_input] #reverseing input seq for better performance

training_output = short_target_token[:round(data_size*(80/100))]



# We will use the remaining for validation

validation_input = short_source_token[round(data_size*(80/100)):]

validation_input  = [val_input[::-1] for val_input in validation_input] #reverseing input seq for better performance

validation_output = short_target_token[round(data_size*(80/100)):]



print('training size', len(training_input))

print('validation size', len(validation_input))
# word encoding for dictionary

vocab = {}

for question in short_source_token:

    for word in question:

        if word not in vocab:

            vocab[word] = 1

        else:

            vocab[word] += 1



for answer in short_target_token:

    for word in answer:

        if word not in vocab:

            vocab[word] = 1

        else:

            vocab[word] += 1  
#remove rare words

threshold = 3

count = 0

for k,v in vocab.items():

    if v >= threshold:

        count += 1

print("Size of total vocab:", len(vocab))

print("Size of vocab we will use:", count)
# creating dictionaries

word_code_start=1

word_code_padding=0





word_num=2 # position 1 is left for word_code_start

encoding = {}

decoding = {1 : 'START'}

for word, count in vocab.items():

    if count>=threshold:

        encoding[word] = word_num

        decoding[word_num]= word

        word_num+=1

print("No. of vocab used : ", word_num)
#word not in dictionary makes as <UNK>

decoding[len(encoding)+2] = '<UNK>'

encoding['<UNK>'] = len(encoding)+2
dict_size=word_num+1
#converting into vector

def transform_into_vector(encoding, data, vector_size=20):

    transformed_data=np.zeros(shape=(len(data),vector_size))

    for i in range(len(data)):

        for j in range(min(len(data[i]),vector_size)):

            try:

                transformed_data[i][j] = encoding[data[i][j]]

            except:

                transformed_data[i][j] = encoding['<UNK>']

    return transformed_data
INPUT_LENGTH = 20

OUTPUT_LENGTH = 20

encoded_training_input = transform_into_vector(

    encoding, training_input, vector_size=INPUT_LENGTH)

encoded_training_output = transform_into_vector(

    encoding, training_output, vector_size=OUTPUT_LENGTH)



print('encoded_training_input', encoded_training_input.shape)

print('encoded_training_output', encoded_training_output.shape)
encoded_training_input[:5]
#encoding validation dataset

encoded_validation_input = transform_into_vector(encoding, validation_input,vector_size=INPUT_LENGTH)

encoded_validation_output = transform_into_vector(encoding, validation_output,vector_size=OUTPUT_LENGTH)

print('Encoded_validation_input',encoded_validation_input.shape)

print('ENcoded_validation_output',encoded_validation_output.shape)
import tensorflow as tf

tf.keras.backend.clear_session()
INPUT_LENGTH=20

OUTPUT_LENGTH=20

encoder_input = Input(shape=(INPUT_LENGTH,))

decoder_input= Input(shape=(OUTPUT_LENGTH,))

print(encoder_input.shape)

print(decoder_input.shape)

from keras.layers import SimpleRNN

encoder = Embedding(dict_size, 128, input_length=INPUT_LENGTH,mask_zero=True)(encoder_input)

encoder = LSTM(512, return_sequences=True, unroll=True)(encoder)

encoder_last=encoder[:,-1,:]



print('Encoder',encoder)

print('Encoder_last', encoder_last)



decoder = Embedding(dict_size,128, input_length=OUTPUT_LENGTH,mask_zero=True)(decoder_input)

decoder = LSTM(512, return_sequences=True, unroll=True)(decoder,initial_state=[encoder_last, encoder_last])

from keras.layers import Activation, dot, concatenate



# Equation (7) with 'dot' score from Section 3.1 in the paper.

# Note that we reuse Softmax-activation layer instead of writing tensor calculation

attention = dot([decoder, encoder], axes=[2, 2])

attention = Activation('softmax', name='attention')(attention)

print('attention', attention)



context = dot([attention, encoder], axes=[2,1])

print('context', context)



decoder_combined_context = concatenate([context, decoder])

print('decoder_combined_context', decoder_combined_context)



# Has another weight + tanh layer as described in equation (5) of the paper

output = TimeDistributed(Dense(512, activation="tanh"))(decoder_combined_context)

output = TimeDistributed(Dense(dict_size, activation="softmax"))(output)

print('output', output)
model = Model(inputs=[encoder_input,decoder_input],outputs=[output])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary
training_encoder_input = encoded_training_input

training_decoder_input = np.zeros_like(encoded_training_output)

training_decoder_input[:, 1:] = encoded_training_output[:,:-1]

training_decoder_input[:, 0] = word_code_start

training_decoder_output = np.eye(dict_size)[encoded_training_output.astype('int')]



validation_encoder_input = encoded_validation_input

validation_decoder_input = np.zeros_like(encoded_validation_output)

validation_decoder_input[:, 1:] = encoded_validation_output[:,:-1]

validation_decoder_input[:, 0] = word_code_start

validation_decoder_output = np.eye(dict_size)[encoded_validation_output.astype('int')]
print(training_encoder_input.shape)

print(training_decoder_output.shape)

print(training_decoder_input.shape)

print(validation_encoder_input.shape)

print(validation_decoder_input.shape)

print(validation_decoder_output.shape)
model.fit(x=[training_encoder_input, training_decoder_input], y=[training_decoder_output],

          validation_data=([validation_encoder_input, validation_decoder_input], [validation_decoder_output]),

          #validation_split=0.1,

          batch_size=128, epochs=100)



from keras.models import load_model



model.save('model_attention_enc_dec.h5')  # creates a HDF5 file 'my_model.h5'

del model  # deletes the existing model
from keras.models import load_model

model = load_model('model_attention_enc_dec.h5')
#model testing

def prediction(raw_text):

    clean_input = clean_text(raw_text)

    input_token = [nltk.word_tokenize(clean_input)]

    input_token = [input_token[0][::-1]] #reversing input token

    encoder_input = transform_into_vector(encoding, input_token, 20)

    decoder_input = np.zeros(shape=(len(encoder_input),OUTPUT_LENGTH))

    decoder_input[:,0] = word_code_start

    for i in range(1, OUTPUT_LENGTH):

        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)

        decoder_input[:,i] = output[:,i]

    return output

def decode(decoding, vector):

    

    text = ''

    for i in vector:

        if i == 0:

            break

        text += ' '

        text += decoding[i]

    return text
for i in range(100):

    seq_index = np.random.randint(1, len(short_source))

    output = prediction(short_source[seq_index])

    print ('Query:', short_source[seq_index])

    print ('Bot:', decode(decoding, output[0]))
'''raw_input = input()

output = prediction(raw_input)

print (decode(decoding, output[0]))

'''