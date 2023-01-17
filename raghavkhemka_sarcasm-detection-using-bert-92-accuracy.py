import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import tensorflow as tf

import matplotlib.pyplot as plt

from transformers import TFBertModel, BertTokenizer

from sklearn.model_selection import train_test_split

data = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json", lines=True)

labels = data.is_sarcastic.values

sentences = data.headline.values

data.head()
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME,do_lower_case = True)



def encoder(sentences):

  ids = []

  for sentence in sentences:

    encoding = tokenizer.encode_plus(

    sentence,

    max_length=16,

    truncation = True,

    add_special_tokens=True,

    return_token_type_ids=False,

    pad_to_max_length=True,

    return_attention_mask=False)

    ids.append(encoding['input_ids'])

  return ids



#Train test split

train_sents,test_sents, train_labels, test_labels  = train_test_split(sentences,labels,test_size=0.15)



train_ids = encoder(train_sents)

test_ids = encoder(test_sents) 
train_ids = tf.convert_to_tensor(train_ids)

test_ids = tf.convert_to_tensor(test_ids)

test_labels = tf.convert_to_tensor(test_labels)

train_labels = tf.convert_to_tensor(train_labels)
bert_encoder = TFBertModel.from_pretrained('bert-base-uncased')

input_word_ids = tf.keras.Input(shape=(16,), dtype=tf.int32, name="input_word_ids")  

embedding = bert_encoder([input_word_ids])

dense = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(embedding[0])

dense = tf.keras.layers.Dense(128, activation='relu')(dense)

dense = tf.keras.layers.Dropout(0.2)(dense)   

output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)    



model = tf.keras.Model(inputs=[input_word_ids], outputs=output)  

model.compile(tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
history = model.fit(x = train_ids, y = train_labels, epochs = 3, verbose = 1, batch_size = 32, validation_data = (test_ids, test_labels))
def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.plot(history.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()



plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')