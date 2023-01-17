# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data =[]

with open('../input/Sarcasm_Headlines_Dataset.json','r') as f:

    for line in f:

        data.append(json.loads(line))
headline = []

labels = []

url = []



for item in data:

        headline.append(item['headline'])

        labels.append(item['is_sarcastic'])

        url.append(item['article_link'])

        
training_size =int(0.5*len(data))



train_sentences = headline[:training_size]

test_sentences = headline[training_size:]



train_labels = labels[:training_size]

test_labels = labels[training_size:]
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
max_length = 32

vocab = 10000

embedding_dim = 16

tunc_type = 'post'

tokenizer = Tokenizer(num_words=vocab,oov_token='<OOV>')
tokenizer.fit_on_texts(train_sentences)
train_seq = tokenizer.texts_to_sequences(train_sentences)

train_padded_seq = tf.keras.preprocessing.sequence.pad_sequences(train_seq,maxlen=max_length,padding='pre',truncating=tunc_type)



test_seq = tokenizer.texts_to_sequences(test_sentences)

test_padded_seq = tf.keras.preprocessing.sequence.pad_sequences(test_seq,maxlen=max_length,padding='pre',truncating=tunc_type)
tf.keras.backend.clear_session()

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab,embedding_dim,input_length = max_length),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(24,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])

model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
num_epoch = 10

model.fit(train_padded_seq,train_labels,epochs=num_epoch,validation_data=(test_padded_seq,test_labels),verbose=2)
e = model.layers[0]
weights = e.get_weights()[0]
weights.shape
word_index = tokenizer.word_index
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
len(reverse_word_index)
import io



out_v = io.open('vecs.tsv', 'w', encoding='utf-8')

out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for word_num in range(1, vocab):

    word = reverse_word_index[word_num]

    embeddings = weights[word_num]

    out_m.write(word + "\n")

    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")

out_v.close()

out_m.close()