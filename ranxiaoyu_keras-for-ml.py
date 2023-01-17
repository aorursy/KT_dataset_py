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
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
import seaborn as sns

target = train['target']

sns.countplot(target)

train.drop(['target'], inplace =True,axis =1)
def concat_df(train, test):

    # Returns a concatenated df of training and test set on axis 0

    return pd.concat([train, test], sort=True).reset_index(drop=True)

df_all = concat_df(train, test)

print(train.shape)

print(test.shape)

print(df_all.shape)

df_all.head()
max([len(t) for t in df_all['text']])
features = ['keyword','location']

for feat in features : 

    print("The number of missing values in "+ str(feat)+" is "+str(df_all[feat].isnull().sum())+ " for the combined dataset")

    print("The number of missing values in "+ str(feat)+" is "+str(train[feat].isnull().sum())+ " for the train dataset")

    print("The number of missing values in "+ str(feat)+" is "+str(test[feat].isnull().sum())+ " for the test dataset")
keyw_train = train['keyword'].unique()

keyw_test = test['keyword'].unique()

print(set(keyw_train)==set(keyw_test))
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



sentences = train['text']

# 80% of total data

train_size = int(7613*0.8)

train_sentences = sentences[:train_size]

train_labels = target[:train_size]



test_sentences = sentences[train_size:]

test_labels = target[train_size:]



# Setting our parameters for the tokenizer (currently using default, we will tune them once we have optimised the rest of the model)

vocab_size = 10000

embedding_dim = 128

max_length = 256

trunc_type='post'

oov_tok = "<OOV>"



tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(train_sentences)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(train_sentences)

padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)



testing_sequences = tokenizer.texts_to_sequences(test_sentences)

testing_padded = pad_sequences(testing_sequences,maxlen=max_length)
import tensorflow as tf

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),

    tf.keras.layers.GlobalAveragePooling1D(),

#     tf.keras.layers.Dense(64, activation='relu'),

#     tf.keras.layers.Dense(64, activation='relu'),

#     tf.keras.layers.Dense(32, activation='relu'),

    

    tf.keras.layers.Dense(16, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(16, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
num_epochs = 10

history = model.fit(padded, train_labels, epochs=num_epochs, validation_data=(testing_padded, test_labels))
tokenizer_1 = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tokenizer_1.fit_on_texts(train['text'])

word_index = tokenizer_1.word_index

sequences = tokenizer_1.texts_to_sequences(train['text'])

padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)



true_test_sentences = test['text']

testing_sequences = tokenizer_1.texts_to_sequences(true_test_sentences)

testing_padded = pad_sequences(testing_sequences,maxlen=max_length)
num_epochs = 8

history = model.fit(padded, target, epochs=num_epochs, verbose=2)
# Now let us deal with testing data

output = model.predict(testing_padded)

pred_plot =  pd.DataFrame(output, columns=['target'])

pred_plot.plot.hist()
final_output = []

for val in pred_plot.target:

    if val > 0.5:

        final_output.append(1)

    else:

        final_output.append(0)
submission['target'] = final_output

# submission['id'] = test['id']

submission.to_csv("final.csv", index=False)

submission.head()