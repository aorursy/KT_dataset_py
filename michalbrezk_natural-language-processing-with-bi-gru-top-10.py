import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, Dense, GRU, Dropout, Bidirectional, SpatialDropout1D

from tensorflow.keras.utils import to_categorical
df_train = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip', sep='\t', usecols=['Phrase', 'Sentiment'])

df_submission = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip', sep='\t', usecols=['Phrase'])
X_train, X_test, y_train, y_test = train_test_split(df_train['Phrase'].values, df_train['Sentiment'].values, test_size=0.1)
# initialize Tokenizer to encode strings into integers

tokenizer = Tokenizer()



# calculate number of rows in our dataset

num_rows = df_train.shape[0]



# create vocabulary from all words in our dataset for encoding

tokenizer.fit_on_texts(df_train['Phrase'].values)



# max length of 1 row (number of words)

row_max_length = max([len(x.split()) for x in df_train['Phrase'].values])



# count number of unique words

vocabulary_size = len(tokenizer.word_index) + 1



# convert words into integers

X_train_tokens = tokenizer.texts_to_sequences(X_train)

X_test_tokens = tokenizer.texts_to_sequences(X_test)

X_sub_tokens = tokenizer.texts_to_sequences(df_submission['Phrase'].values)



# ensure every row has same size - pad missing with zeros

X_train_pad = pad_sequences(X_train_tokens, maxlen=row_max_length, padding='post')

X_test_pad = pad_sequences(X_test_tokens, maxlen=row_max_length, padding='post')

X_sub_pad = pad_sequences(X_sub_tokens, maxlen=row_max_length, padding='post')
y_train_cat = to_categorical(y_train)

y_test_cat = to_categorical(y_test)



target_length = y_train_cat.shape[1]

print('Original vector size: {}'.format(y_train.shape))

print('Converted vector size: {}'.format(y_train_cat.shape))
EMBEDDING_DIM = 256



model = Sequential()

model.add(Embedding(vocabulary_size, EMBEDDING_DIM, input_length=row_max_length))

model.add(SpatialDropout1D(0.2))

model.add(Bidirectional(GRU(128)))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(target_length, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)

history = model.fit(X_train_pad, y_train_cat, epochs=5, validation_data=(X_test_pad, y_test_cat), batch_size=128, callbacks=[callback])
# predict test data

y_sub_hat_ = model.predict(X_sub_pad)

y_sub_hat = [np.argmax(x) for x in y_sub_hat_]



# save to csv

df_save = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv')

df_save['Sentiment'] = y_sub_hat

df_save.to_csv('Submission.csv', index = False)

print('Submission saved!')