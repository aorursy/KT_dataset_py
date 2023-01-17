# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset=pd.read_csv('../input/hin.txt',sep='\t',header=None)
dataset.columns=['english','hindi']

dataset['english']=dataset['english'].apply(lambda x:x+' $')

dataset['hindi']=dataset['hindi'].apply(lambda x:'^ '+x+' $')

dataset.head()
X=dataset.english

Y=dataset.hindi
X_words_counts={}

for row in X:

    for words in row.split(' '):

        #print(words)

        X_words_counts[words]=X_words_counts.get(words,0)+1
most_common_X_words = sorted(X_words_counts.items(), key=lambda x: x[1], reverse=True)[:3]

print(most_common_X_words)
Y_words_counts={}

for row in Y:

    for words in row.split(' '):

        #print(words)

        Y_words_counts[words]=Y_words_counts.get(words,0)+1
most_common_Y_words = sorted(Y_words_counts.items(), key=lambda x: x[1], reverse=True)[:3]

print(most_common_Y_words)
cnt=0

X_WORDS_TO_INDEX={}

for w in X_words_counts:

    X_WORDS_TO_INDEX[w] =cnt

    cnt+=1

X_WORDS_TO_INDEX['#']=len(X_WORDS_TO_INDEX)

ALL_X_WORDS = X_WORDS_TO_INDEX.keys()

print(X_WORDS_TO_INDEX)
cnt=0

Y_WORDS_TO_INDEX={}

for w in Y_words_counts:

    Y_WORDS_TO_INDEX[w] =cnt

    cnt+=1

Y_WORDS_TO_INDEX['#']=len(Y_WORDS_TO_INDEX)

ALL_Y_WORDS = Y_WORDS_TO_INDEX.keys()

print(Y_WORDS_TO_INDEX)
def length_of_sentence(sentence):

    return len(sentence.split(' '))

dataset['e_length']=dataset['english'].apply(length_of_sentence)

dataset['h_length']=dataset['hindi'].apply(length_of_sentence)

Tx=dataset.e_length.max()

Ty=dataset.h_length.max()

print(Tx)

print(Ty)
def series_to_array(series,vocab):

    X_train=[]

    for row in series:

        r=row.split(' ')

        R=[]

        for a in r:

            R.append(vocab[a])

        X_train.append(R)

    length = max(map(len, X_train))

    y=np.array([xi+[None]*(length-len(xi)) for xi in X_train])

    return y
X_train=series_to_array(dataset.english,X_WORDS_TO_INDEX)

X_train=np.where(X_train==None, len(X_WORDS_TO_INDEX)-1, X_train)

print(X_train)
from keras.utils import to_categorical

Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(X_WORDS_TO_INDEX)), X_train)))

print(Xoh.shape)
Y_train=series_to_array(dataset.hindi,Y_WORDS_TO_INDEX)

Y_train=np.where(Y_train==None, len(Y_WORDS_TO_INDEX)-1, Y_train)

print(Y_train)

Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(Y_WORDS_TO_INDEX)), Y_train)))

Yo=Yoh[:,1:,:]

print(Yo.shape)

ze=np.zeros((1,len(Y_WORDS_TO_INDEX)))

ze[0][Y_WORDS_TO_INDEX['^']]=1

Yo=np.insert(arr=Yo,obj=Ty-1,values=ze,axis=1)

print(Yo.shape)
n_a=2048

from keras.models import Sequential

from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply,LSTMCell,RNN,BatchNormalization

from keras.layers import RepeatVector, Dense, Activation, Lambda, Reshape,TimeDistributed,Concatenate

from keras.optimizers import Adam

from keras.utils import to_categorical

from keras.models import load_model, Model

import keras.backend as K

from keras import metrics
encoder_inputs = Input(shape=(Tx, len(X_WORDS_TO_INDEX)))

encoder = Bidirectional(LSTM(n_a, return_state=True))

encoder_outputs, state_h_f, state_c_f,state_h_b,state_c_b = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.

state_h=Concatenate()([state_h_f,state_h_b])

state_c=Concatenate()([state_c_f,state_c_b])

print(state_h.shape)

encoder_states = [state_h, state_c]



# Set up the decoder, using `encoder_states` as initial state.

decoder_inputs = Input(shape=(None, len(Y_WORDS_TO_INDEX)))

# We set up our decoder to return full output sequences,

# and to return internal states as well. We don't use the 

# return states in the training model, but we will use them in inference.

decoder_lstm = LSTM(2*n_a, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(decoder_inputs,

                                     initial_state=encoder_states)

decoder_dense = TimeDistributed(Dense(len(Y_WORDS_TO_INDEX), activation='softmax'))

decoder_outputs = decoder_dense(decoder_outputs)

print(decoder_outputs.shape)



# Define the model that will turn

# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()
opt = Adam(lr=0.001,beta_1=0.9,beta_2=0.999,decay=0.01)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit([Xoh,Yoh], Yo, epochs=300, batch_size=64)
encoder_model = Model(encoder_inputs, encoder_states)



decoder_state_input_h = Input(shape=(2*n_a,))

decoder_state_input_c = Input(shape=(2*n_a,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(

    decoder_inputs, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(

    [decoder_inputs] + decoder_states_inputs,

    [decoder_outputs] + decoder_states)
INDEX_TO_WORD_Y={y:x for (x,y) in Y_WORDS_TO_INDEX.items()}
def decode_sequence(input_seq):

    # Encode the input as state vectors.

    states_value = encoder_model.predict(input_seq)



    # Generate empty target sequence of length 1.

    target_seq = np.zeros((1, 1, len(Y_WORDS_TO_INDEX)))

    # Populate the first character of target sequence with the start character.

    target_seq[0, 0, Y_WORDS_TO_INDEX['^']] = 1



    # Sampling loop for a batch of sequences

    # (to simplify, here we assume a batch of size 1).

    stop_condition = False

    decoded_sentence = ''

    while not stop_condition:

        output_tokens, h, c = decoder_model.predict(

            [target_seq] + states_value)



        # Sample a token

        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        sampled_char = INDEX_TO_WORD_Y[sampled_token_index]

        decoded_sentence += sampled_char



        # Exit condition: either hit max length

        # or find stop character.

        if (sampled_char == '$'):

            stop_condition = True



        # Update the target sequence (of length 1).

        target_seq = np.zeros((1, 1, len(Y_WORDS_TO_INDEX)))

        target_seq[0, 0, sampled_token_index] = 1.



        # Update states

        states_value = [h, c]



    return decoded_sentence



cunt=0

for index, row in dataset.iterrows():

    source=Xoh[index,:,:]

    source=source.reshape(1,Tx,len(X_WORDS_TO_INDEX))

    #print(source)

    result=decode_sequence(source)

    print('##########')

    print("source:", row['english'])

    print('real:',row['hindi'])

    print("output:", result)

    if cunt==10:

        break

    cunt+=1