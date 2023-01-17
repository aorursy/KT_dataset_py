# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import tensorflow as tf





from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

''' for checking if there are any missing values in the dataset:

for col in train:

    print (train[col].isnull().sum())

    

There are no missing values in the dataset as seen

    

'''

'''

#number of insincere questions in our dataset

insin=len(train.loc[train['target']==1])

sin=len(train.loc[train['target']==0])

insin/len(train)

#So only 6 percent of the dataset comprises of insincere questions

'''
train_df, test_df = train_test_split(train, test_size=0.2)

tr_sentences = train_df['question_text'].tolist()

tr_labels = train_df['target'].tolist()



te_sentences = test_df['question_text'].tolist()

te_labels = test_df['target'].tolist()
tokenizer = Tokenizer(num_words=95000, oov_token="<OOV>")

tokenizer.fit_on_texts(tr_sentences)



tr_sequences = tokenizer.texts_to_sequences(tr_sentences)

tr_padded = pad_sequences(tr_sequences,maxlen=100, padding = 'post', truncating='pre')



te_sequences = tokenizer.texts_to_sequences(te_sentences)

te_padded = pad_sequences(te_sequences, maxlen=100, padding='post', truncating = 'pre')

print('done')
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(95000, 300, input_length = 100),

    tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(128, return_sequences=True)),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dropout(0.2), 

    tf.keras.layers.Dense(32, activation = 'relu'), 

    tf.keras.layers.Dropout(0.2), 

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

model.summary()
history = model.fit(tr_padded, tr_labels, epochs=2, batch_size=512, validation_data=(te_padded, te_labels), verbose=1)
#Thanks to : https://www.kaggle.com/advaitsave/lstm-using-tensorflow-2-with-embeddings

from sklearn import metrics

pred_test_y = model.predict([te_padded], batch_size=1024, verbose=1)



opt_prob = None

f1_max = 0



for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    f1 = metrics.f1_score(te_labels, (pred_test_y > thresh).astype(int))

    print('F1 score at threshold {} is {}'.format(thresh, f1))

    

    if f1 > f1_max:

        f1_max = f1

        opt_prob = thresh

        

print('Optimal probabilty threshold is {} for maximum F1 score {}'.format(opt_prob, f1_max))
sentences = train['question_text'].tolist()

labels = train['target'].tolist()

test_submission=test['question_text'].tolist()



tokenizer_final = Tokenizer(num_words=95000, oov_token="<OOV>")

tokenizer_final.fit_on_texts(sentences)



sequences = tokenizer.texts_to_sequences(sentences)

test_submission = tokenizer.texts_to_sequences(test_submission)



padded = pad_sequences(sequences,maxlen=100, padding = 'post', truncating='pre')

test_submission = pad_sequences(test_submission, maxlen=100, padding = 'post', truncating='pre')

print('done')



history = model.fit(padded, labels, epochs=2, batch_size=512, verbose=1)







pred_submission = model.predict([test_submission], batch_size=1024, verbose=1)

pred_submission = (pred_submission > opt_prob).astype(int)



df_submission = pd.DataFrame({'qid': test['qid'].values})

df_submission['prediction'] = pred_submission

df_submission.to_csv("submission.csv", index=False)