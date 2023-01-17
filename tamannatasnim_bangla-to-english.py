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
import string

import re

from numpy import array, argmax, random, take

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense, LSTM, Embedding, RepeatVector

from keras.preprocessing.text import Tokenizer

from keras.callbacks import ModelCheckpoint

from keras.preprocessing.sequence import pad_sequences

from keras.models import load_model

from keras import optimizers

import matplotlib.pyplot as plt

%matplotlib inline

pd.set_option('display.max_colwidth', 200)
# function to read raw text file

def read_text(filename):

        # open the file

        file = open(filename, mode='rt', encoding='utf-32')

        

        # read all text

        text = file.read()

        file.close()

        return text
# split a text into sentences

def to_lines(text):

      sents = text.strip().split('\n')

      sents = [i.split('\t') for i in sents]

      return sents
data = read_text("/kaggle/input/bangla/ben.txt")

ben_eng = to_lines(data)

ben_eng = array(ben_eng)
ben_eng = ben_eng[:50000,:]
ben_eng
# Remove punctuation

ben_eng[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in ben_eng[:,0]]

ben_eng[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in ben_eng[:,1]]



ben_eng
# convert text to lowercase

for i in range(len(ben_eng)):

    ben_eng[i,0] = ben_eng[i,0].lower()

    ben_eng[i,1] = ben_eng[i,1].lower()



ben_eng
# empty lists

eng_l = []

ben_l = []



# populate the lists with sentence lengths

for i in ben_eng[:,0]:

      eng_l.append(len(i.split()))



for i in ben_eng[:,1]:

      ben_l.append(len(i.split()))



length_df = pd.DataFrame({'eng':eng_l, 'ben':ben_l})



length_df.hist(bins = 30)

plt.show()
# function to build a tokenizer

def tokenization(lines):

      tokenizer = Tokenizer()

      tokenizer.fit_on_texts(lines)

      return tokenizer
# prepare english tokenizer

eng_tokenizer = tokenization(ben_eng[:, 0])

eng_vocab_size = len(eng_tokenizer.word_index) + 1



eng_length = 8

print('English Vocabulary Size: %d' % eng_vocab_size)
# prepare bangla tokenizer

ben_tokenizer = tokenization(ben_eng[:, 1])

ben_vocab_size = len(ben_tokenizer.word_index) + 1



ben_length = 8

print('bangla Vocabulary Size: %d' % ben_vocab_size)
# encode and pad sequences

def encode_sequences(tokenizer, length, lines):

         # integer encode sequences

         seq = tokenizer.texts_to_sequences(lines)

         # pad sequences with 0 values

         seq = pad_sequences(seq, maxlen=length, padding='post')

         return seq
from sklearn.model_selection import train_test_split



# split data into train and test set

train, test = train_test_split(ben_eng, test_size=0.2, random_state = 12)
# prepare training data

trainX = encode_sequences(ben_tokenizer, ben_length, train[:, 1])

trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])



# prepare validation data

testX = encode_sequences(ben_tokenizer, ben_length, test[:, 1])

testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
# build NMT model

def define_model(in_vocab,out_vocab, in_timesteps,out_timesteps,units):

      model = Sequential()

      model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))

      model.add(LSTM(units))

      model.add(RepeatVector(out_timesteps))

      model.add(LSTM(units, return_sequences=True))

      model.add(Dense(out_vocab, activation='softmax'))

      return model
# model compilation

model = define_model(ben_vocab_size, eng_vocab_size, ben_length, eng_length, 512)
rms = optimizers.RMSprop(lr=0.001)

model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')
filename = 'model.h1.24_jan_19'

checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')



# train model

history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1),

                    epochs=30, batch_size=512, validation_split = 0.2,callbacks=[checkpoint], 

                    verbose=1)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['train','validation'])

plt.show()
model = load_model('model.h1.24_jan_19')

preds = model.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))
def get_word(n, tokenizer):

      for word, index in tokenizer.word_index.items():

          if index == n:

              return word

      return None
preds_text = []

for i in preds:

       temp = []

       for j in range(len(i)):

            t = get_word(i[j], eng_tokenizer)

            if j > 0:

                if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):

                     temp.append('')

                else:

                     temp.append(t)

            else:

                   if(t == None):

                          temp.append('')

                   else:

                          temp.append(t) 



       preds_text.append(' '.join(temp))
pred_df = pd.DataFrame({'actual' : test[:,0], 'predicted' : preds_text})
pred_df.sample(15)
# load doc into memory

def load_doc(filename):

	# open the file as read only

	file = open(filename, mode='rt', encoding='utf-8')

	# read all text

	text = file.read()

	# close the file

	file.close()

	return text
# split a loaded document into sentences

def to_pairs(doc):

	lines = doc.strip().split('\n')

	pairs = [line.split('\t') for line in  lines]

	return pairs
# clean a list of lines

def clean_pairs(lines):

	cleaned = list()

	# prepare regex for char filtering

	re_print = re.compile('[^%s]' % re.escape(string.printable))

	# prepare translation table for removing punctuation

	table = str.maketrans('', '', string.punctuation)

	for pair in lines:

		clean_pair = list()

		for line in pair:

			# normalize unicode characters

			line = normalize('NFD', line).encode('ascii', 'ignore')

			line = line.decode('UTF-8')

			# tokenize on white space

			line = line.split()

			# convert to lowercase

			line = [word.lower() for word in line]

			# remove punctuation from each token

			line = [word.translate(table) for word in line]

			# remove non-printable chars form each token

			line = [re_print.sub('', w) for w in line]

			# remove tokens with numbers in them

			line = [word for word in line if word.isalpha()]

			# store as string

			clean_pair.append(' '.join(line))

		cleaned.append(clean_pair)

	return array(cleaned)
import string

import re

from pickle import dump

from unicodedata import normalize

from numpy import array



# load doc into memory

def load_doc(filename):

	# open the file as read only

	file = open(filename, mode='rt', encoding='utf-8')

	# read all text

	text = file.read()

	# close the file

	file.close()

	return text



# split a loaded document into sentences

def to_pairs(doc):

	lines = doc.strip().split('\n')

	pairs = [line.split('\t') for line in  lines]

	return pairs



# clean a list of lines

def clean_pairs(lines):

	cleaned = list()

	# prepare regex for char filtering

	re_print = re.compile('[^%s]' % re.escape(string.printable))

	# prepare translation table for removing punctuation

	table = str.maketrans('', '', string.punctuation)

	for pair in lines:

		clean_pair = list()

		for line in pair:

			# normalize unicode characters

			line = normalize('NFD', line).encode('ascii', 'ignore')

			line = line.decode('UTF-8')

			# tokenize on white space

			line = line.split()

			# convert to lowercase

			line = [word.lower() for word in line]

			# remove punctuation from each token

			line = [word.translate(table) for word in line]

			# remove non-printable chars form each token

			line = [re_print.sub('', w) for w in line]

			# remove tokens with numbers in them

			line = [word for word in line if word.isalpha()]

			# store as string

			clean_pair.append(' '.join(line))

		cleaned.append(clean_pair)

	return array(cleaned)



# save a list of clean sentences to file

def save_clean_data(sentences, filename):

	dump(sentences, open(filename, 'wb'))

	print('Saved: %s' % filename)



# load dataset

filename = '/kaggle/input/bangla/ben.txt'

doc = load_doc(filename)

# split into english-german pairs

pairs = to_pairs(doc)

# clean sentences

clean_pairs = clean_pairs(pairs)

# save clean pairs to file

save_clean_data(clean_pairs, 'english-german.pkl')

# spot check

for i in range(100):

	print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))
dataset=pd.read_csv('/kaggle/input/bangla/ben.txt',sep='\t',header=None)
dataset.columns=['english','bangla','set']

dataset['english']=dataset['english'].apply(lambda x:x+' $')

dataset['bangla']=dataset['bangla'].apply(lambda x:'^ '+x+' $')

dataset.head()
X=dataset.english

Y=dataset.bangla
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

dataset['b_length']=dataset['bangla'].apply(length_of_sentence)

Tx=dataset.e_length.max()

Ty=dataset.b_length.max()

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
Y_train=series_to_array(dataset.bangla,Y_WORDS_TO_INDEX)

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

model.fit([Xoh,Yoh], Yo, epochs=3, batch_size=64)

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