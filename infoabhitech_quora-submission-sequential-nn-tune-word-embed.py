# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from numpy import array

from numpy import asarray

from numpy import zeros



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Dropout

from keras.layers import Bidirectional

from keras.layers import CuDNNGRU

from keras.layers.embeddings import Embedding
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
# Using Keras Tokenizer to fit on training data



tokenizer = Tokenizer()

tokenizer.fit_on_texts(df_train['question_text'])
vocab_size = len(tokenizer.word_index) + 1
# Using Keras function to convert text token to number sequence



ts_train=tokenizer.texts_to_sequences(df_train['question_text'])
ts_test=tokenizer.texts_to_sequences(df_test['question_text'])
# Padding number sequence to a length that could be token size of a longest question

# Small questions will have zero's at the end



X_train_vectorized=pad_sequences(ts_train,maxlen=135,padding='post')
X_test_vectorized=pad_sequences(ts_test,maxlen=135,padding='post')
y_train = df_train['target']
# Using only Glove word embedding out of other 3 provided for this competition(Google , Paragram , Wiki)

# Glove considered to have slight better accuracy and same could be checked with resulting F1 accuracy/score

# Word embedding has token and corresponding 300 weight features , we will load file entire in memory



embeddings_index = {}

f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt', encoding='utf8')

for line in f:

    values = line.split()

    word = ''.join(values[:-300])

    coefs = np.asarray(values[-300:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('Loaded %s word vectors.' % len(embeddings_index))
# Weight matrix is created for only those tokens present in our question corpus



embedding_matrix = zeros((vocab_size, 300))

for word, i in tokenizer.word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
# Sequential Neural netwok model with weights as embedding

# Model parameters (output dimension ,dropout, activation function have been manually tuned for better output)



model = Sequential()

e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=135, trainable=False)

model.add(e)

model.add(Bidirectional(LSTM(128, return_sequences=True)))

model.add(Flatten())

model.add(Dense(128, activation='sigmoid'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())
# Model is trained on the vectorized token data as input and label as output

# Less epochs has been used , higher may overfit and may not generalize well and can bring less score

# High batch size will help in faster training



model.fit(X_train_vectorized, y_train, epochs=3, batch_size=1024, verbose=0)
# Resulting prediction will come in float , but we have to transform it to integer for 0/1 labelling.



predictions = model.predict(X_test_vectorized)
# Float value greater than 0.33 is converted to integer 1 and below to 0

# 0.33 gives better result than 0.5 if used for 0 and 1 distinction 



preds_class = (predictions > 0.33).astype(np.int)
# Reshaping size of prediction array so as to conver it to series



preds=preds_class.reshape(len(df_test),)
prediction = pd.Series(preds,name="prediction")
qid = df_test['qid']
# Concatting qid and prediction for submission



submission_df = pd.concat([qid, prediction], axis=1)
submission_df.head()
submission_df.to_csv("submission.csv", columns = submission_df.columns, index=False)