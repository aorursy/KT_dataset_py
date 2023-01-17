import numpy as np
import pandas as pd

from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.preprocessing import text, sequence
from keras import regularizers

from collections import Counter
hyper_params = {
    'validation_split': 0.1,
    'batch_size': 32,
    'epochs': 10,
    'embedding_size': 128,
    'keep_probability': 0.9,
    'lstm_size': 50,
    'dense_size': 50,
    'max_sequence': 100
}
# Download from https://www.kaggle.com/c/8076/download/train.csv.zip
train = pd.read_csv('../input/train.csv')
train_sentences = train.comment_text.values

# Download from https://www.kaggle.com/c/8076/download/test.csv.zip
test = pd.read_csv('../input/test.csv')
test_sentences = test.comment_text.values

CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(np.concatenate((train_sentences, test_sentences), axis=0))
train_tokenized = tokenizer.texts_to_sequences(train_sentences)
test_tokenized = tokenizer.texts_to_sequences(test_sentences)
vocab_size = len(Counter([token for sublist in train_tokenized + test_tokenized for token in sublist])) + 1

X_train = sequence.pad_sequences(train_tokenized, maxlen=hyper_params['max_sequence'])
y_train = train[CLASSES].values
X_test = sequence.pad_sequences(test_tokenized, maxlen=hyper_params['max_sequence'])
model = Sequential()

model.add(Embedding(vocab_size, hyper_params['embedding_size']))
model.add(LSTM(hyper_params['lstm_size']))
model.add(Dropout(1 - hyper_params['keep_probability']))
model.add(Dense(hyper_params['dense_size'], activation='relu'))
model.add(Dropout(1 - hyper_params['keep_probability']))
model.add(Dense(len(CLASSES), activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, batch_size=hyper_params['batch_size'], epochs=hyper_params['epochs'], 
                    validation_split=hyper_params['validation_split'])
# Download from https://www.kaggle.com/c/8076/download/sample_submission.csv.zip
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission[CLASSES] = model.predict(X_test)
sample_submission.to_csv('submission.csv', index=False)