# Libraries

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split



import gc,warnings

warnings.filterwarnings("ignore")



import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, Dense, GRU, Dropout, Bidirectional, SpatialDropout1D

from tensorflow.keras.utils import to_categorical



#Gensim Library for Text Processing

import gensim.parsing.preprocessing as gsp

from gensim import utils
# Load Data



df_train = pd.read_csv('../input/nlp-getting-started/train.csv', header='infer')

df_test = pd.read_csv('../input/nlp-getting-started/test.csv', header='infer')

df_submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv', header='infer')



print("Total Records(train): ",df_train.shape[0])

print("Total Records(test): ",df_test.shape[0])

# Drop Unwanted columnms



unwanted_cols = ['keyword','location']

df_train.drop(unwanted_cols,axis=1,inplace=True)

df_test.drop(unwanted_cols,axis=1,inplace=True)
'''Text Cleaning Utility Function'''



processes = [

               gsp.strip_tags, 

               gsp.strip_punctuation,

               gsp.strip_multiple_whitespaces,

               gsp.strip_numeric,

               gsp.remove_stopwords, 

               gsp.strip_short

            ]



# Utility Function

def clean_txt(txt):

    text = txt.lower()

    text = utils.to_unicode(text)

    for p in processes:

        text = p(text)

    return text



# Applying the function to text column

df_train['text'] = df_train['text'].apply(lambda x: clean_txt(x))

df_test['text'] = df_test['text'].apply(lambda x: clean_txt(x))

# Data Split

X_train, X_test, y_train, y_test = train_test_split(df_train['text'].values, df_train['target'].values, test_size=0.1)
# Garbage Collect

gc.collect()
# initialize Tokenizer to encode strings into integers

tokenizer = Tokenizer()



# calculate number of rows in our dataset

num_rows = df_train.shape[0]





# create vocabulary from all words in our dataset for encoding

tokenizer.fit_on_texts(df_train['text'].values)



# max length of 1 row (number of words)

row_max_length = max([len(x.split()) for x in df_train['text'].values])



# count number of unique words

vocabulary_size = len(tokenizer.word_index) + 1



# convert words into integers

X_train_tokens = tokenizer.texts_to_sequences(X_train)

X_test_tokens = tokenizer.texts_to_sequences(X_test)



# ensure every row has same size - pad missing with zeros

X_train_pad = pad_sequences(X_train_tokens, maxlen=row_max_length, padding='post')

X_test_pad = pad_sequences(X_test_tokens, maxlen=row_max_length, padding='post')

'''Data Preparation for Test Data'''



# initialize Tokenizer to encode strings into integers

tok_tst = Tokenizer()



# calculate number of rows in our dataset

num_rows_tst = df_test.shape[0]





# create vocabulary from all words in our dataset for encoding

tok_tst.fit_on_texts(df_test['text'].values)



# max length of 1 row (number of words)

row_max_length_tst = max([len(x.split()) for x in df_test['text'].values])



# count number of unique words

vocab_size_tst = len(tok_tst.word_index) + 1



# convert words into integers

X_TST_tokens = tok_tst.texts_to_sequences(df_test['text'])



# ensure every row has same size - pad missing with zeros

X_TST_pad = pad_sequences(X_TST_tokens, maxlen=row_max_length_tst, padding='post')
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

model.add(Dense(128, activation='sigmoid'))

model.add(Dropout(0.2))

model.add(Dense(target_length, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])



callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)

history = model.fit(X_train_pad, y_train_cat, epochs=5, validation_data=(X_test_pad, y_test_cat), batch_size=64, callbacks=[callback])
# Evaluation

results = model.evaluate(X_test_pad, y_test_cat, batch_size=128, verbose=0)

print("Model Accuracy: ",'{:.2%}'.format(results[1]))
# Garbage Collection

gc.collect()
# Making Prediction

y_pred = model.predict(X_TST_pad)
# Copying the predicted target to submission

df_submission['target'] = np.round(y_pred).astype('int')
# save to csv

df_submission.to_csv('Submission.csv', index = False)

print('Submission saved!')