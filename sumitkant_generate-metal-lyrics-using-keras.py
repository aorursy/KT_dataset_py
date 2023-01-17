import os

import glob

import numpy as np
BAND_NAME = 'MACHINE HEAD'

BASE_PATH =  '../input/large-metal-lyrics-archive-228k-songs/metal_lyrics'
lyrics_files = glob.glob(os.path.join(BASE_PATH, BAND_NAME.lower()[0], BAND_NAME,'*','*.txt'))



# printing a sample

with open(lyrics_files[0]) as f: 

    print (f.read(200))

    

# generating corpus

corpus = []

for lyric in lyrics_files:

    with open(lyric) as f: 

        corpus.append(f.read())



# cleaning up the lyrics

corpus = '\n'.join([x.lower() for x in corpus])       # join all songs to form on big corpus

corpus = corpus.split('\n')                           # split the lines

corpus = [x for x in corpus if not x.startswith('[')] # removing comments starting with [

corpus = [x for x in corpus if x != '']               # removing empty items

corpus[:10]                                           # get first 25 lines
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer()

tokenizer.fit_on_texts(corpus)

word_index = tokenizer.word_index

reverse_word_index = dict([(v,k) for (k,v) in word_index.items()])



TOTAL_WORDS = len(word_index) + 1

print ('Total Words :', TOTAL_WORDS) # added 1 for OOV token
input_sequences = []

for line in corpus:

    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):

        n_gram_seq = token_list[:i+1]

        input_sequences.append(n_gram_seq)
MAX_SEQ_LEN = max([len(x) for x in input_sequences])

print ('Max Sequence Length :', MAX_SEQ_LEN)
from tensorflow.keras.preprocessing.sequence import pad_sequences

input_sequences = np.array(pad_sequences(input_sequences, maxlen = MAX_SEQ_LEN, padding = 'pre'))



xs = input_sequences[:,:-1]

labels = input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes = TOTAL_WORDS)

xs.shape, ys.shape
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout

model = Sequential([

    Embedding(TOTAL_WORDS, 256, input_length = MAX_SEQ_LEN - 1),

    Bidirectional(LSTM(128, return_sequences = True)),

    Bidirectional(LSTM(128)),

    Dropout(0.2),

    Dense(TOTAL_WORDS, activation = 'softmax')

])



model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

model.summary()

%time history = model.fit(xs, ys, epochs = 200, verbose = 0, batch_size = 256)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.title('LOSS')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.show()



plt.plot(history.history['accuracy'])

plt.title('ACCURACY')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.show()
seed_text  = 'Let freedom ring with a shotgun blast'

next_words = 50

for _ in range(next_words):

    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    token_list = pad_sequences([token_list], maxlen = MAX_SEQ_LEN - 1, padding = 'pre')

    predicted  = model.predict_classes(token_list, verbose = 0)

    seed_text += ' ' + reverse_word_index[predicted[0]]

print (seed_text)
character_corpus = '\n'.join(corpus)

character_corpus[:100]

VOCAB = list(set(character_corpus))

VOCAB_SIZE = len(VOCAB) + 1

TOTAL_SIZE = len(character_corpus)

word_index = dict([(x,i+1) for (i,x) in enumerate(VOCAB)]) 

reverse_word_index = dict([(v,k) for (k,v) in word_index.items()])
print ('TOTAL SIZE :', TOTAL_SIZE)

print ('VOCAB SIZE :', VOCAB_SIZE)
print ([x for x in word_index.keys()])
xs = []; ys = []

SEQ_LEN = 30

for i in range(TOTAL_SIZE - SEQ_LEN):

    xs.append([word_index[x] for x in character_corpus[i:i+SEQ_LEN]])

    ys.append(word_index[character_corpus[i+SEQ_LEN]])
xs = np.array(xs)

ys = np.array(tf.keras.utils.to_categorical(ys, num_classes = VOCAB_SIZE))

xs.shape, ys.shape
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, Conv1D, MaxPooling1D

model = Sequential([

    Embedding(VOCAB_SIZE, VOCAB_SIZE, input_length = SEQ_LEN),

    Conv1D(32, (3), activation = 'relu'),

    Conv1D(32, (3), activation = 'relu'),

    MaxPooling1D(2),

    Bidirectional(LSTM(64, return_sequences = True)),

    Bidirectional(LSTM(32)),

    Dropout(0.2),

    Dense(VOCAB_SIZE, activation = 'softmax')

])



model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

model.summary()

%time history = model.fit(xs, ys, epochs = 200, verbose = 0, batch_size = 1024)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.title('LOSS')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.show()



plt.plot(history.history['accuracy'])

plt.title('ACCURACY')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.show()
seed_text  = 'let freedom ring with a shotgun blast'

next_words = 500

for _ in range(next_words):

    seed_reshaped = np.array([word_index[x] for x in seed_text[-SEQ_LEN:]]).reshape(1, SEQ_LEN)

    predicted  = model.predict_classes(seed_reshaped, verbose = 0)

    seed_text += reverse_word_index[predicted[0]]

print (seed_text)