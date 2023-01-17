# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing import sequence, text
from keras.layers import Embedding, Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten


data = pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
data.head()




data['sentiment'].value_counts()


def remove_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def clean_text(raw_text):
    text = remove_html(raw_text)
    return text
data['review'] = data['review'].apply(clean_text)
data.head()
data['sentiment'].value_counts()
# maximum number of words to keep, based on word frequency
vocab_size = 10000

tokenizer = text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(data['review'])
sequences = tokenizer.texts_to_sequences(data['review'])
word_index = tokenizer.word_index
# maximum length of all sequences
max_len = 100

x = sequence.pad_sequences(sequences, maxlen=max_len)
sentiments = {
    'positive': 1,
    'negative': 0
}

y = np.asarray(data['sentiment'].map(sentiments))


train_samples = 40000

x_train = x[:train_samples]
y_train = y[:train_samples]

x_test = x[train_samples:]
y_test = y[train_samples:]


def load_glove(path):
    
    embedding_index = {}
    for line in open(path):
        values = line.split()
        word = values[0]
        coeff = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coeff
    
    return embedding_index


embedding_index = load_glove('../input/datasettxt/glove.6B.100d.txt')


embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, index in word_index.items():
    if index < vocab_size:
        vector = embedding_index.get(word);
        if vector is not None:
            embedding_matrix[index] = embedding_index.get(word)

print('Shape of embedding matrix:', embedding_matrix.shape)

def get_LSTM_model(units = 32, dropout = 0):
    
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len, weights=[embedding_matrix], trainable=False),
        LSTM(units, dropout=dropout),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    
    return model


def get_GRU_model(units = 32, dropout = 0):
    
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len, weights=[embedding_matrix], trainable=False),
        GRU(units, dropout=dropout),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    
    return model


def get_CNN_model(filters = 32, filter_size = 7, pool_size = 5):
    
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len, weights=[embedding_matrix], trainable=False),
        Conv1D(filters, filter_size, activation='relu'),
        MaxPooling1D(pool_size),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    
    return model
LSTM_model = get_LSTM_model()
LSTM_model.summary()
GRU_model = get_GRU_model()
GRU_model.summary()


CNN_model = get_CNN_model()
CNN_model.summary()


epochs = 5
batch_size = 32
val_split = 0.2
LSTM_history = LSTM_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=val_split)
LSTM_model.save('LSTM_imdb_sentiment_analysis.h5')
GRU_history = GRU_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=val_split)
GRU_model.save('GRU_imdb_sentiment_analysis.h5')
CNN_history = CNN_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=val_split)
CNN_model.save('CNN_imdb_sentiment_analysis.h5')

def plot_graph(history, title = 'accuracy and loss graphs'):
    
    acc_values = history.history['acc']
    val_acc_values = history.history['val_acc']

    loss_values = history.history['loss']
    val_loss_values = history.history['val_loss']

    epochs_range = range(1, epochs + 1)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title)
    
    ax[0].plot(epochs_range, acc_values, label='Training accuracy')
    ax[0].plot(epochs_range, val_acc_values, label='Validation accuracy')
    ax[0].set(xlabel='Epochs', ylabel='Accuracy')
    ax[0].legend()
    ax[0].set_title('Accuracy')

    ax[1].plot(epochs_range, loss_values, label='Training loss')
    ax[1].plot(epochs_range, val_loss_values, label='Validation loss')
    ax[1].set(xlabel='Epochs', ylabel='Loss')
    ax[1].legend()
    ax[1].set_title('Loss')



plot_graph(LSTM_history, 'LSTM Model')
plot_graph(GRU_history, 'GRU Model')
plot_graph(CNN_history, 'CNN Model')


def test_model(model):
    scores = model.evaluate(x_test, y_test)
    print('Loss: {}'.format(scores[0]))
    print('Accuracy: {}'.format(scores[1]))
print('LSTM Model')
test_model(LSTM_model)
print('GRU Model')
test_model(GRU_model)
print('CNN Model')
test_model(CNN_model)
