import copy
import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as sk_train_test_split

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
def get_categories(df):
    return df['category'].unique()
# Load the Keras tokenizer
# Note that it will use only the most "num_words" used words
def load_tokenizer(X_data, num_words=150000):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X_data)
    return tokenizer
def data_to_sequences(X_data, tokenizer, max_sequence_length):
    X_data = tokenizer.texts_to_sequences(X_data)
    X_data = sequence.pad_sequences(X_data, maxlen=max_sequence_length)
    return X_data
def train_test_split(X_data, Y_data, tokenizer, max_sequence_length):
    X_data = data_to_sequences(X_data, tokenizer, max_sequence_length)
    
    Y_data = Y_data.astype(np.int32)
    X_train, X_test, Y_train, Y_test = sk_train_test_split(X_data, Y_data, test_size=0.3)
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    return X_train, X_test, Y_train, Y_test
df = pd.read_csv('../input/bbc-articles-cleaned/tfidf_dataset.csv')
df.head()
X_data = df[['text']].to_numpy().reshape(-1)
Y_data = df[['category']].to_numpy().reshape(-1)
category_to_id = {}
category_to_name = {}

for index, c in enumerate(Y_data):
    if c in category_to_id:
        category_id = category_to_id[c]
    else:
        category_id = len(category_to_id)
        category_to_id[c] = category_id
        category_to_name[category_id] = c
    
    Y_data[index] = category_id

# Display dictionary
category_to_name
MAX_SEQUENCE_LENGTH = 1000

n_texts = len(X_data)
print('Texts in dataset: %d' % n_texts)

n_categories = len(get_categories(df))
print('Number of categories: %d' % n_categories)

print('Loading tokenizer...')
tokenizer = load_tokenizer(X_data)

print('Loading train dataset...')
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, tokenizer, MAX_SEQUENCE_LENGTH)

print('Done!')
def load_embedding_matrix(tokenizer):
    embedding_dim = 100
    embeddings_index = {}

    f = open('../input/glove6b/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix, embedding_dim
def create_lstm_model(tokenizer, input_length, n_categories):
    word_index = tokenizer.word_index
    embedding_matrix, embedding_dim = load_embedding_matrix(tokenizer)

    model = Sequential()
    model.add(Embedding(input_dim=len(word_index) + 1,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=input_length,
                        trainable=True))
    model.add(SpatialDropout1D(0.3))
    model.add(LSTM(64,
                   activation='tanh',
                   dropout=0.2,
                   recurrent_dropout=0.5))
    model.add(Dense(n_categories, activation='softmax'))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model
EPOCHS = 10

model = create_lstm_model(tokenizer, MAX_SEQUENCE_LENGTH, n_categories)
history = model.fit(X_train,
                    Y_train,
                    epochs=EPOCHS,
                    validation_data=(X_test, Y_test),
                    verbose=1)
def plot_confusion_matrix(X_test, Y_test, model):
    Y_pred = model.predict_classes(X_test)
    con_mat = tf.math.confusion_matrix(labels=Y_test, predictions=Y_pred).numpy()

    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    label_names = list(range(len(con_mat_norm)))

    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=label_names, 
                              columns=label_names)

    figure = plt.figure(figsize=(10, 10))
    sns.heatmap(con_mat_df, cmap=plt.cm.Blues, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

x_labels = range(1, EPOCHS + 1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(x_labels, acc, color='b', linestyle='-', label='Training acc')
plt.plot(x_labels, val_acc, color='b', linestyle='--', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_labels, loss, color='b', linestyle='-', label='Training acc')
plt.plot(x_labels, val_loss, color='b', linestyle='--', label='Validation acc')
plt.title('Training and validation loss')
plt.legend()

plt.show()
plot_confusion_matrix(X_test, Y_test, model)
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
category_to_name
