# Imports
import numpy as np
import pandas as pd
from subprocess import check_output
import matplotlib.pyplot as plt
%matplotlib inline
print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/better-donald-trump-tweets/Donald-Tweets!.csv')
print('Rows', data.shape[0])
print('Colummns', data.shape[1])
data.head()
text = '\n\n'.join(data['Tweet_Text'].values)
from collections import Counter
import re
cntr = Counter(text)
rare = list(np.asarray(list(cntr.keys()))[np.asarray(list(cntr.values())) < 300])
for c in rare:
    text = re.sub('[' + c + ']', '', text)
text[:1000]
print('Total de Caracteres no Corpus:', len(text))
chars = sorted(list(set(text)))
print('Total de Caracteres Únicos:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen = 50
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('Número de Sequências:', len(sentences))
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
import random
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
cntr = Counter(text)
cntr_sum = sum(cntr.values())
char_probs = list(map(lambda c: cntr[c] / cntr_sum, chars))
def sample(preds):
    preds = np.asarray(preds).astype('float64')
    preds = preds / np.sum(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
def generate(model, length, seed=''):
    
    if len(seed) != 0:
        sys.stdout.write(seed)
    
    generated = seed
    sentence = seed
    
    for i in range(length):
        x = np.zeros((1, maxlen, len(chars)))

        padding = maxlen - len(sentence)
        
        for i in range(padding):
            x[0, i] = char_probs # pad usando os anteriores
            
        for t, char in enumerate(sentence):
            x[0, padding + t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds)
        next_char = indices_char[next_index]

        sentence = sentence[1:] + next_char
        generated += next_char
        
        sys.stdout.write(next_char)
        sys.stdout.flush()
        
    return generated
from os.path import isfile
from keras.models import load_model

MODEL_PATH = '../input/weights/stacked-lstm-2-layers-128-hidden.h5'

if isfile(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    N_HIDDEN = 128

    # Modelo
    model = Sequential()
    model.add(LSTM(N_HIDDEN, dropout = 0.1, input_shape = (maxlen, len(chars)), return_sequences = True))
    model.add(LSTM(N_HIDDEN, dropout = 0.1))
    model.add(Dense(len(chars), activation = 'softmax'))

    # Otimizador
    optimizer = RMSprop(lr = 0.01)
    
    # Compilação
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)

    # Imprime amostras a cada época
    for iteration in range(1, 40):
        print('\n')
        print('-' * 4)
        print('\nIteração', iteration)
        model.fit(X, y, batch_size=3000, epochs=1)

        print('\n-------------------- Tweet Gerado Pelo Modelo Nesta Iteração ---------------------\n')

        rand = np.random.randint(len(text) - maxlen)
        seed = text[rand:rand + maxlen]
        generate(model, 400, seed)

    #model.save(MODEL_PATH)
sample_tweet_start = 'Go Republican Senators, Go!'
_ = generate(model, 250, sample_tweet_start)
