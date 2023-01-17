# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pickle

import string

import warnings

import numpy as np

import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!wget https://github.com/amarlearning/neural-machine-translation/raw/master/model/weights-improvement.hdf5
data = pd.read_csv('/kaggle/input/frenchenglish-translation/fra.tsv', delimiter='\t')

data.head()
data = data.iloc[:55000, :]



english = data.english.values

french = data.french.values
print("Length of english sentence:", len(english))

print("Length of french sentence:", len(french))

print('-'*20)

print(english[100])

print('-'*20)

print(french[100])
english = [s.translate(str.maketrans('', '', string.punctuation)) for s in english]

french = [s.translate(str.maketrans('', '', string.punctuation)) for s in french]



print(english[100])

print('-'*20)

print(french[100])
english = [s.lower() if isinstance(s, str) else s for s in english]

french = [s.lower() if isinstance(s, str) else s for s in french]



print(english[100])

print('-'*20)

print(french[100])
eng_l = [len(s.split()) for s in english]

fre_l = [len(s.split()) for s in french]



length_df = pd.DataFrame({'english': eng_l, 'french': fre_l})

length_df.hist(bins=30)

plt.show()
from keras import optimizers

from keras.models import Sequential, load_model

from keras.preprocessing.text import Tokenizer

from keras.utils.vis_utils import plot_model

from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import ModelCheckpoint

from keras.layers import Dense, Embedding, LSTM, RepeatVector, Dropout, Bidirectional, Flatten
def tokenizer(corpus):

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(corpus)

    return tokenizer



english_tokenizer = tokenizer(english)

french_tokenizer = tokenizer(french)



word_index_english = english_tokenizer.word_index

word_index_french = french_tokenizer.word_index



eng_vocab_size = len(word_index_english) + 1

fre_vocab_size = len(word_index_french) + 1
print("Size of english vocab:", len(word_index_english))

print("Size of french vocab:", len(word_index_french))
max_len_eng = max(eng_l)

max_len_fre = max(fre_l)



print("Max length of english sentence:", max_len_eng)

print("Max length of french sentence:", max_len_fre)
english = pd.Series(english).to_frame('english')

french = pd.Series(french).to_frame('french')



dummy_df = pd.concat([english, french], axis=1)

train, test = train_test_split(dummy_df, test_size=0.07, random_state=42)



train_english = train.english.values

train_french = train.french.values



test_english = test.english.values

test_french = test.french.values
def encode_sequences(tokenizer, length, text):

    sequences = tokenizer.texts_to_sequences(text)

    sequences = pad_sequences(sequences, maxlen=length, padding='post')

    return sequences
eng_seq = encode_sequences(english_tokenizer, max_len_eng, train_english)

fre_seq = encode_sequences(french_tokenizer, max_len_fre, train_french)



# test_english = encode_sequences(english_tokenizer, max_len_eng, test_english)

test_french = encode_sequences(french_tokenizer, max_len_fre, test_french)



print(eng_seq[10])

print(fre_seq[10])
# saving

with open('english_tokenizer.pickle', 'wb') as handle:

    pickle.dump(english_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

# saving

with open('french_tokenizer.pickle', 'wb') as handle:

    pickle.dump(french_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
! ls
def nmt_model(in_vocab_size, out_vocab_size, in_timestep, out_timestep, units):

    model = Sequential()

    model.add(Embedding(in_vocab_size, 100, input_length=in_timestep, mask_zero=True))

    model.add(Bidirectional(LSTM(units, dropout=0.5, recurrent_dropout=0.4)))

    model.add(Dropout(0.5))

    model.add(RepeatVector(out_timestep))

    model.add(Bidirectional(LSTM(units, dropout=0.5, recurrent_dropout=0.4, return_sequences=True)))

    model.add(Dropout(0.5))

    model.add(Dense(out_vocab_size, activation="softmax"))

    return model
model = nmt_model(fre_vocab_size, eng_vocab_size, max_len_fre, max_len_eng, 256)
rms = optimizers.RMSprop(lr=0.001)

model.compile(loss="sparse_categorical_crossentropy", optimizer=rms, metrics=['accuracy'])
model.summary()
plot_model(model, show_shapes=True)
model = load_model('weights-improvement.hdf5')
filepath="weights-improvement.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
eng_seq = eng_seq.reshape(eng_seq.shape[0], eng_seq.shape[1], 1)

history = model.fit(fre_seq, eng_seq, batch_size=1024, epochs=20, verbose=1, validation_split=0.05, shuffle=True, callbacks=[checkpoint])
# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
prediction = model.predict_classes(test_french.reshape(test_french.shape[0], test_french.shape[1]))
def get_word(n, tokenizer):

    for word, index in tokenizer.word_index.items():

        if index == n:

            return word

    return None
preds_text = []

for i in tqdm(prediction):

    temp = []

    for j in range(len(i)):

        t = get_word(i[j], english_tokenizer)

        if j > 0:

            if (t == get_word(i[j-1], english_tokenizer)) or (t == None):

                temp.append('')

            else:

                temp.append(t)

        else:

            if(t == None):

                temp.append('')

            else:

                temp.append(t)

    preds_text.append(' '.join(temp))
pred_df = pd.DataFrame({'actual' : test_english, 'predicted' : preds_text})
pred_df.head(7)
pred_df.tail(7)