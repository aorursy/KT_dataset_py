import pandas as pd
import numpy as np
import keras
import os
print(os.listdir("../input"))
train2 = pd.read_csv('../input/cell-me/train.csv')
train2 = test.replace(np.nan, '', regex=True)
#Dane dostępne, zawierające dane testowe jak i treningowe konkursu
df = pd.read_csv('../input/all-data/Amazon_Unlocked_Mobile.csv')
df = df.replace(np.nan, '', regex=True)
#Pobranie danych do uczenia i testowania
train = df[:202646]
test = df[202646:]
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
texts = df['Reviews']
from sklearn.utils import shuffle
train = shuffle(train)
x_train = train['Reviews']
# dzielemy, aby uzyskać dane z przedziału 0-1
y_train = train['Rating']/5
x_test = test['Reviews']
# dzielemy, aby uzyskać dane z przedziału 0-1
y_test = test['Rating']/5
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
#Sprawdzenia średnią długość rezencji (koło 193)
print(sum( map(len, x_train) ) / len(x_train))

maxlen = 200
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)
#Wynikiem jest sekwencja słowo i jej index
x_train = pad_sequences(sequences, maxlen=maxlen)
# Ucinamy jeśli większe od 200 długość recenzji, jak krótsze dopisujemy zera
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
sequences = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(sequences, maxlen=maxlen)
embeddings_index = {}
f = open('../input/twitter-data/glove.twitter.27B.100d.txt')
# Reprezentacja wektorowa słów
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))
word_index = tokenizer.word_index
# Pobranie słownika ze wszystkimi unikatowymi słowami/tokenami i ich index
word_index
embedding_dim = 100
#Ograniczenie rozmiaru repezentacji słowa przez wektor
embedding_matrix = np.zeros((max_words, embedding_dim))
#Zbudowanie vektor z vectorami samych zer o dłogości 100
# wierzemy wszytskie nauczone się tokeny i dla tych których mamy w twitter GloVe przypisujemy jego wektorową reprezentację
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
from keras.models import Model
from keras.layers import Embedding, Flatten, Dense, Dropout, GRU, CuDNNGRU, Bidirectional, CuDNNLSTM, Input, LSTM, Concatenate
from keras import regularizers
import tensorflow as tf
from keras.models import Sequential
from keras.layers import  GaussianNoise,Embedding, Flatten, Dense, Dropout, GRU, CuDNNGRU, CuDNNLSTM, Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate, concatenate
from keras import regularizers
input_layer = Input(shape=(maxlen,))
# Sieć dwukierunkowa dla poprawy interpretacji słowa (od początku i końca recenzji patrząc)
model = Sequential()
x = Embedding(max_words, embedding_dim, input_length=maxlen)(input_layer)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = Dropout(0.4)(x)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = Bidirectional(CuDNNGRU(32))(x)
x = Dense(32, activation='relu')(x)
output_layer = Dense(1)(x)
model = Model(inputs=input_layer, outputs=output_layer)
# Ustawienie wartość dla pierwszej warstwy, aby uczyć się tylko tych wyrazów które mamy z recenzji
print(model.layers[1].get_weights())
print([embedding_matrix])
model.layers[1].set_weights([embedding_matrix])
model.layers[1].trainable = False
model.summary()
model.compile(optimizer='rmsprop',loss='mse',metrics=['mse'])
history = model.fit(x_train, y_train, epochs=8, batch_size=256, validation_data=(x_test, y_test), verbose=2)
model.save('0.0249')
df2 = pd.read_csv('../input/cell-me/test.csv')

x_test2 = df2['Reviews']
x_test2= x_test2.astype(str)
sequences = tokenizer.texts_to_sequences(x_test2)
sequences = tokenizer.texts_to_sequences(x_test2)
x_test2 = pad_sequences(sequences, maxlen=maxlen)
from keras.models import load_model
model = load_model('0.0249')
pred = model.predict(x_test)
pred
yy_test = y_test*5
ppred = pred*5
#Mnożymy w celu powrotu do wartości 1-5
from sklearn.metrics import mean_squared_error
ppred = ppred.flatten()
#Spłaszczamy wynik do jednej tablicy
mean_squared_error(ppred,yy_test)
dat = pd.DataFrame()
ppredd = np.clip(ppred,1,5)
#zaokrąglamy,aby nie przekroczyć skali oceny 1-5
res = test.drop(['Product Name', 'Brand Name','Price' , 'Rating', 'Reviews','Review Votes'], axis=1)
res['Rating'] = ppredd
res
res.to_csv('Result5.csv',index=False)