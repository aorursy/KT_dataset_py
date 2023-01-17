# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import  pandas as pd
import re
import spacy

#load data
train_data=pd.read_csv('../input/nlp-getting-started/train.csv')
test_data=pd.read_csv('../input/nlp-getting-started/test.csv')
#pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
train_data.drop(["keyword","location"],axis = 1,inplace = True)
train_data.head()

#clean Data/preprocessing 
train_data["text"] = train_data["text"].apply(lambda x : " ".join(x.lower() for x in x.split()))
train_data["text"] = train_data["text"].str.replace("\d","")
train_data["text"] = train_data["text"].str.replace("[^\w\s]","")
train_data.head()
#convert data into list to feed the neurons
train_data_text=train_data['text'].tolist()
test_data_text=test_data['text'].tolist()
train_target=train_data['target'].tolist()
print(len(train_data_text))
#split data for training 80:20
training_size=int(len(train_data_text)*0.8)

training_text=train_data_text[0:training_size]
training_target=train_target[0:training_size]
testing_text=train_data_text[training_size:]
testing_target=train_target[training_size:]

training_targets_final=np.array(training_target)
testing_targets_final=np.array(testing_target)

print(len(training_targets_final))

print(len(testing_targets_final))
#Create and train model
vocab_size=1000
embedding_dim=16
max_len=100
trunc_type='post'
padding_type='post'

tokenizer=Tokenizer(num_words=vocab_size,oov_token="<OOV>")
tokenizer.fit_on_texts(training_text)
word_index=tokenizer.word_index

training_text=tokenizer.texts_to_sequences(training_text)
training_padding=pad_sequences(training_text,maxlen=max_len,padding=padding_type,truncating=trunc_type)

testing_text=tokenizer.texts_to_sequences(testing_text)
testing_padding=pad_sequences(testing_text,maxlen=max_len,padding=padding_type,truncating=trunc_type)

model=tf.keras.Sequential([
tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_len),
tf.keras.layers.GlobalAveragePooling1D(),
tf.keras.layers.Dense(6,activation='relu'),
tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history=model.fit(training_padding, training_targets_final, epochs=30, validation_data=(testing_padding, testing_targets_final))
#plot prediction
import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
