import numpy as np 

import pandas as pd

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import classification_report, f1_score

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation

from keras.layers.embeddings import Embedding
VOCAB_SIZE = 20000

MAX_LEN = 100

NUM_CLASSES = 6
train_df = pd.read_csv('../input/train.csv', index_col='id').fillna(' ')

val_df = pd.read_csv('../input/valid.csv', index_col='id').fillna(' ')

test_df = pd.read_csv('../input/test.csv', index_col='id').fillna(' ')
label_binarizer = LabelBinarizer().fit(pd.concat([train_df['label'], val_df['label']]))
y_train = label_binarizer.transform(train_df['label'])

y_valid = label_binarizer.transform(val_df['label'])
X_train, X_valid = train_df['text'].values, val_df['text'].values
vocabulary_size = VOCAB_SIZE

tokenizer = Tokenizer(num_words=vocabulary_size)

tokenizer.fit_on_texts(X_train)

sequences = tokenizer.texts_to_sequences(X_train)

X_train = pad_sequences(sequences, maxlen=MAX_LEN)
sequences = tokenizer.texts_to_sequences(X_valid)

X_valid = pad_sequences(sequences, maxlen=MAX_LEN)
model = Sequential()

model.add(Embedding(VOCAB_SIZE, 100, input_length=MAX_LEN))

model.add(Dropout(rate=0.5))

model.add(Conv1D(64, 5, activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(100))

model.add(Dense(NUM_CLASSES, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train,

                    batch_size=1024,

                    epochs=10,

                    verbose=1,

                    validation_split=0.1)
score = model.evaluate(X_valid, y_valid, batch_size=256, verbose=1)

print('Validation accuracy:', score[1])
val_preds = model.predict(X_valid)
print(classification_report(np.argmax(y_valid,axis=1), np.argmax(val_preds, axis=1)))
f1_score(np.argmax(y_valid,axis=1), np.argmax(val_preds, axis=1), average='micro')
!git clone https://github.com/jasonwei20/eda_nlp.git

!rm -rf eda_nlp/.git
train_df['sent_len'] = train_df['text'].apply(lambda s: len(s.split()))



train_df = train_df[train_df['sent_len'] > 1]
train_df.head()
train_df[['label', 'text']].to_csv('train.tsv', sep='\t', index=None, header=None)
%%time

!python eda_nlp/code/augment.py --input=train.tsv --output=train_aug.tsv
!wc -l train_aug.tsv
train_df_aug = pd.read_csv('train_aug.tsv', sep='\t', header=None,

                          names=['label', 'text'])
train_df_aug.head()
X_train_aug = train_df_aug['text'].values
y_train_aug = label_binarizer.transform(train_df_aug['label'])
%%time

vocabulary_size = VOCAB_SIZE

tokenizer = Tokenizer(num_words=vocabulary_size)

tokenizer.fit_on_texts(X_train_aug)

sequences = tokenizer.texts_to_sequences(X_train_aug)

X_train_aug = pad_sequences(sequences, maxlen=MAX_LEN)
sequences = tokenizer.texts_to_sequences(val_df['text'])

X_valid = pad_sequences(sequences, maxlen=MAX_LEN)
model_aug = Sequential()

model_aug.add(Embedding(VOCAB_SIZE, 100, input_length=MAX_LEN))

model_aug.add(Dropout(rate=0.5))

model_aug.add(Conv1D(64, 5, activation='relu'))

model_aug.add(MaxPooling1D(pool_size=2))

model_aug.add(LSTM(100))

model_aug.add(Dense(NUM_CLASSES, activation='softmax'))
model_aug.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_aug.fit(X_train_aug, y_train_aug,

          batch_size=1024, epochs=3,

          verbose=1, validation_split=0.1)
score = model_aug.evaluate(X_valid, y_valid, batch_size=256, verbose=1)

print('Validation accuracy:', score[1])
val_preds = model_aug.predict(X_valid)
print(classification_report(np.argmax(y_valid,axis=1), np.argmax(val_preds, axis=1)))
f1_score(np.argmax(y_valid,axis=1), np.argmax(val_preds, axis=1), average='micro')