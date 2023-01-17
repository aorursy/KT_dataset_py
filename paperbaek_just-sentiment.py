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
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
#training_size = 20000
encoder = LabelEncoder()
encoder.fit(train_df['sentiment'])
sentiment = encoder.transform(train_df['sentiment'])
labels = tf.keras.utils.to_categorical(sentiment, num_classes=3)
x_train = train_df['text']

sentences = []

for l in x_train:
    sentences.append(l)
sentences = np.array(sentences)
labels = np.array(labels)
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')

tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, truncating=trunc_type, padding=padding_type, maxlen=max_length)
model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint_path = 'tweet_model_checkpoint.ckpt'
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    monitor='val_loss',
                                                    verbose=1)

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=3,
                                                verbose=1,
                                                mode='auto')
history = model.fit(padded, labels,
                        validation_split=0.2,
                        epochs=20,
                        callbacks=[checkpoint, earlystopping])
model.load_weights(checkpoint_path)
model.save("tweet_model.h5")
test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
test_df
x_test = test_df['text']

test_sentences = []

for l in x_test:
    test_sentences.append(l)
    
test_sentences = np.array(test_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

test_padded = pad_sequences(test_sequences, truncating=trunc_type, padding=padding_type, maxlen=max_length)
encoder = LabelEncoder()
encoder.fit(test_df['sentiment'])
sentiment = encoder.transform(test_df['sentiment'])
test_labels = tf.keras.utils.to_categorical(sentiment, num_classes=3)
sub = model.evaluate(test_padded, test_labels)
#answer = np.argmax(sub, axis=-1)