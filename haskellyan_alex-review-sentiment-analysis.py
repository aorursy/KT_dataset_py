import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os

import keras
from keras import preprocessing
sns.set()
df = pd.read_csv('../input/amazon_alexa.tsv', sep='\t')
df.head()
vocab_size = 5000
maxlen = 200
def prepare_data(df, vocab_size, maxlen, split_ratio=.7):
    data = df.verified_reviews
    resp = (df.rating > 3).astype('float32')
    
    tokenizer = preprocessing.text.Tokenizer(vocab_size)
    tokenizer.fit_on_texts(data)
    
    # translate words to integers
    tokens = tokenizer.texts_to_sequences(data)
    
    # align the size of reviews, we are prepending zeros if the review is not long enough.
    data = preprocessing.sequence.pad_sequences(tokens, maxlen=maxlen, dtype='float32')
    
    # split the data intro two sets: training, testing.
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    train_size = np.floor(split_ratio * df.shape[0]).astype('int32')
    
    train_index = index[:train_size]
    test_index = index[train_size:]
    
    return tokenizer, data[train_index], resp[train_index], data[test_index], resp[test_index]

tokenizer, X_train, Y_train, X_test, Y_test = prepare_data(df, vocab_size, maxlen)
from keras import layers
from keras import models
from keras import optimizers
model = models.Sequential([
    layers.Embedding(vocab_size, 10, input_length=maxlen),
    layers.BatchNormalization(),
    layers.LSTM(10, dropout=.1, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(.35),
    layers.Dense(1, activation='sigmoid')
])

optimizer = optimizers.Adam(lr=.0015)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
def plot_history(history):
    history = history.history
    
    print(history.keys())
    
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], label='accuracy')
    plt.plot(history['val_acc'], label='val_accuracy')
    plt.legend()
checkpoint = keras.callbacks.ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True)
reduceLROnPlateau = keras.callbacks.ReduceLROnPlateau()
earlyStopping = keras.callbacks.EarlyStopping(patience=2)

history = model.fit(
    X_train, Y_train, validation_split=.2, epochs=10, 
    callbacks=[checkpoint, reduceLROnPlateau, earlyStopping]
)
plot_history(history)
from keras.models import load_model

# Load the best one, as during training our model is a bit "overcooked".
model = load_model('model.h5')
model.evaluate(X_test, Y_test)