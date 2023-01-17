# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train= pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test= pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.info()
train.head()
validation_size=0.2

training_size= int((1-validation_size)*len(train))
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

from tensorflow.keras.optimizers import Adam
import nltk

nltk.download('punkt')

from nltk.tokenize import word_tokenize

import string

import re

import spacy

sp = spacy.load('en_core_web_sm')

from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

lemma= WordNetLemmatizer()
def clean_data(text):

    #remove emails

    text = ' '.join([i for i in text.split() if '@' not in i])

    

    #remove web address

    text = re.sub('http[s]?://\S+', '', text)

    

    #Filter to allow only alphabets

    text = re.sub(r'[^a-zA-Z\']', ' ', text)

    

    #Remove Unicode characters

    text = re.sub(r'[^\x00-\x7F]+', '', text)

    

    #Convert to lowercase to maintain consistency

    text = text.lower()

    

    #Remove stopwords

    all_stopwords = sp.Defaults.stop_words

    text_tokens = word_tokenize(text)

    tokens_without_sw= [word for word in text_tokens if not word in all_stopwords]

      

    #lemmatization

    text= [lemma.lemmatize(w) for w in tokens_without_sw]

    text= ' '.join(text)

  

    #remove double spaces 

    text = re.sub('\s+', ' ',text)

    return (text)
print(train['text'][0])

print(train['text'][1])
train['text']= train['text'].apply(clean_data)
print(train['text'][0])

print(train['text'][1])
train.head()
train=train.reindex(np.random.permutation(train.index))

train= train.reset_index(drop=True)
train.head()
corpus=[]

labels=[]

for i in range(len(train)):

    corpus.append(train['text'][i])

    labels.append(train['target'][i])



print(corpus[2])

print(labels[2])
#vocab_size=20000

oov_token= "<oov>"

padding_type='post'

trunc_type='post'

embedding_dim=100

max_len= max([len(x) for x in corpus])
train_data= corpus[0:training_size]

validation_data= corpus[training_size:]



training_labels=labels[0:training_size]

validation_labels= labels[training_size:]
tokenizer=Tokenizer()

tokenizer.fit_on_texts(corpus)

word_index= tokenizer.word_index

vocab_size= len(word_index)

print(vocab_size)
training_sequences=tokenizer.texts_to_sequences(train_data)

padded_training= pad_sequences(training_sequences,padding=padding_type,maxlen=max_len)



validation_sequences= tokenizer.texts_to_sequences(validation_data)

padded_validation= pad_sequences(validation_sequences,padding=padding_type,truncating=trunc_type,maxlen=max_len)
print(test['text'][0])

print(test['text'][1])
test['text']=test['text'].apply(clean_data)
print(test['text'][0])

print(test['text'][1])
testing_sentences=[]



for i in range(len(test)):

    testing_sentences.append(test['text'][i])

    

testing_sequences= tokenizer.texts_to_sequences(testing_sentences)

padded_testing= pad_sequences(testing_sequences,padding=padding_type,maxlen=max_len)
padded_training= np.array(padded_training)

training_labels= np.array(training_labels)



padded_validation= np.array(padded_validation)

validation_labels= np.array(validation_labels)



padded_testing= np.array(padded_testing)
embeddings_index = {};

GLOVE_DIR= '../input/glove-global-vectors-for-word-representation'

with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:

    for line in f:

        values = line.split();

        word = values[0];

        coefs = np.asarray(values[1:], dtype='float32');

        embeddings_index[word] = coefs;



embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word);

    if embedding_vector is not None:

        embeddings_matrix[i] = embedding_vector;
model= Sequential()

model.add(Embedding(vocab_size+1,embedding_dim,input_length=max_len, weights=[embeddings_matrix],trainable=False))

model.add(Dropout(0.3))

model.add(Bidirectional(LSTM(64,return_sequences= True)))

model.add(Bidirectional(LSTM(32)))

model.add(Dense(32,activation='relu'))

model.add(Dense(1,activation='sigmoid'))
model.summary()
adam= Adam(0.0003)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
history=model.fit(padded_training,training_labels,epochs=20,validation_data=(padded_validation,validation_labels))
import matplotlib.pyplot as plt





def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.plot(history.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()



plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')
predictions= model.predict_classes(padded_testing)
predictions
sample=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
sample.head()
sample['target']= (predictions>0.5).astype(int)
sample.to_csv("new_submission.csv",index=False, header=True)