# Import Libraries
import sys
import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import glob
filepath = "/kaggle/input/donald-trumps-rallies/"
for f in glob.glob(filepath+"*.txt"):
    os.system("cat "+f+" >> DonaldTrumpRallies.txt")
# Load Dataset
filename    = 'DonaldTrumpRallies.txt'
text        = open(filename, encoding='utf-8').read()
text        = text.lower()
print('corpus length:', len(text))

# Find all the unique characters
chars        = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
vocab_size   = len(chars)

print("List of unique characters : \n", chars)
print("Number of unique characters : \n", vocab_size)
print("Character to integer mapping : \n", char_indices)
# Preprocessing Dataset
max_seq_len = 64 # cut text in semi-redundant sequences of max_seq_len characters
step = 3 
sentences = [] # list_X
next_chars= [] # list_Y

for i in range(0, len(text) - max_seq_len, step):
    sentences.append(text[i: i + max_seq_len])
    next_chars.append(text[i + max_seq_len])
print('nb sequences:', len(sentences))

num_sequences  = len(sentences)
print("Number of sequences: ", num_sequences)
print(sentences[0])
print('Vectorization...')
train_X = np.zeros((len(sentences), max_seq_len, len(chars)), dtype=np.bool)
train_Y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        train_X[i, t, char_indices[char]] = 1
    train_Y[i, char_indices[next_chars[i]]] = 1

print(train_X.shape)
print(train_Y.shape)
input_shape = (max_seq_len, vocab_size)
print(input_shape)
# Build Model
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(len(chars), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.summary()
# Train Model
num_epochs = 30
batch_size = 128
#model_path = "BiLSTM-DonaldTrumpRallies .h5"
#checkpoint = ModelCheckpoint(model_path, monitor='loss', save_best_only=True, verbose=1, mode='min')
#callbacks_list = [checkpoint]

model.fit(train_X, train_Y, epochs = num_epochs, batch_size = batch_size, verbose=1) #, callbacks=callbacks_list)
model.save('BiLSTM-DonaldTrumpRallies.h5')
# Generate Text
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text():
    start_index = random.randint(0, len(text) - max_seq_len - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)
        generated = ''
        sentence = text[start_index: start_index + max_seq_len]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, max_seq_len, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

generate_text()
