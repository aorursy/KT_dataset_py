!pip install pymorphy2

import nltk as nltk
import pandas as pd
import numpy as np
import pymorphy2
import tensorflow as tf

!pip install -I tensorflow

from keras.preprocessing.text import Tokenizer
import numpy as np
from keras. preprocessing.sequence import pad_sequences
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv1D, Activation, GlobalMaxPool1D, LSTM
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras import layers
analyzer = pymorphy2.MorphAnalyzer()

def get_normal_form_of_single_text(text):
    normalized_text = ""
    tokens = nltk.word_tokenize(text)
    
    words_array = []
    
    for token in tokens:            
        words_array.append(analyzer.parse(token)[0].normal_form)
    return words_array


def create_embedding_matrix(df, word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for index in range(len(df["word"])):
        if df["word"][index] in word_index:
            idx = word_index[df["word"][index]]
            embedding_matrix[idx] = np.array(
                df["vectors"][index], dtype=np.float32)[:embedding_dim]

    return embedding_matrix
# df = pd.read_csv("../input/allreviews/all-reviews.csv", names=["name", "review", "label"], skiprows=lambda i: i % 10 != 0 or i == 0)
df = pd.read_csv("../input/allreviews/all-reviews.csv", names=["name", "review", "label"], skiprows=1)
df['label'] = df['label'].map(lambda x: "1" if x not in ["-1", "0", "1"] else x)

df['review'] = df['review'].map(lambda x: get_normal_form_of_single_text(x))
print(df['review'])
my_films = ["Побег из Шоушенка", "Поймай меня, если сможешь", "Престиж"]
my_reviews = df[df.name.isin(my_films)]
train_reviews = df[~df.name.isin(my_films)]
rusvectores_df = pd.read_csv('../input/rusvectores/model.txt', skiprows=1, sep=r'\s{2,}', engine='python', names=['line'])
print(rusvectores_df.head())

rusvectores_df = pd.DataFrame(rusvectores_df.line.str.split(r'\s{1,}', 1), columns=['line'])
rusvectores_df = rusvectores_df.line.apply(pd.Series)
rusvectores_df.columns = ['word', 'vectors']
print(rusvectores_df.head())

rusvectores_df.word = rusvectores_df.word.map(lambda word: word.split("_", 1)[0])
rusvectores_df.vectors = rusvectores_df.vectors.map(lambda vectors: vectors.split(" "))
print(rusvectores_df.head())
tokenizer = Tokenizer(num_words=len(rusvectores_df.word))
tokenizer.fit_on_texts(rusvectores_df.word)

print(train_reviews['review'].head())
train_reviews['review'] = tokenizer.texts_to_sequences(train_reviews['review'])
my_reviews['review'] = tokenizer.texts_to_sequences(my_reviews['review'])
print(train_reviews['review'].head())

embedding_dim = 300
maxlen = 500
embedding_matrix = create_embedding_matrix(rusvectores_df, tokenizer.word_index, embedding_dim)

X_train_padded = pad_sequences(train_reviews['review'].to_numpy(), maxlen=maxlen, padding='post')
X_test_padded = pad_sequences(my_reviews['review'].to_numpy(), maxlen=maxlen, padding='post')
y_train = to_categorical(train_reviews['label'], 3)
y_test = to_categorical(my_reviews['label'], 3)
embedding_input = len(tokenizer.word_index) + 1
print(len(tokenizer.word_index) + 1)
model = Sequential()
model.add(Embedding(embedding_input, embedding_dim, weights=[embedding_matrix], input_length=maxlen))
model.add(LSTM(embedding_dim, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='sigmoid'))
model.compile(metrics=["accuracy"], optimizer='adam', loss='binary_crossentropy')
model.summary()
history = model.fit(X_train_padded, y_train, epochs=10, batch_size=32, verbose=False, validation_data=(X_test_padded, y_test))

loss, accuracy = model.evaluate(X_train_padded, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

result = model.predict(X_test_padded)
print(classification_report(y_test.argmax(axis=1), result.argmax(axis=1)))