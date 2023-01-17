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
from transformers import AutoTokenizer, TFBertForSequenceClassification
from nltk.corpus import stopwords, wordnet
from transformers import *
from nltk import download
import tensorflow as tf
from io import open
import numpy as np
import json
import re

#tpu
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# download('stopwords')
download('wordnet')

#Get rid of noise from dataset
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    word_list = string.split(' ')
    string = ""
    for word in word_list:
        if word not in stopwords.words('english'):
            if wordnet.synsets(word):
                string = string + word + " "
    return string.strip().lower()

#Political Bias Stuff - modified from https://github.com/icoen/CS230P/blob/master/RNN/data_helpers2.py
def load_data_and_labels(positive_data_file, negative_data_file):
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]

    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    positive_labels = [[1] for _ in positive_examples]
    negative_labels = [[0] for _ in negative_examples]

    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

#returns tokenized data
def tokenize_sentences(sentences, tokenizer, max_seq_len = 50):
    tokenized_sentences = []
    for sentence in sentences:
        tokenized_sentence = tokenizer.encode(sentence, add_special_tokens = True, max_length = max_seq_len)
        tokenized_sentences.append(tokenized_sentence)
    return tokenized_sentences

def preprocessPoliticalData(dem_file,rep_file):
    print("Loading data...")
    x_text, y = load_data_and_labels(dem_file,rep_file)
    tokenizer =  AutoTokenizer.from_pretrained('bert-base-uncased', do_lowercase=True)

    x_train = tokenize_sentences(x_text, tokenizer)
    x = np.array(list(tf.keras.preprocessing.sequence.pad_sequences(x_train, 50, padding='post', truncating='post')))

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_train = x[shuffle_indices]
    y_train = y[shuffle_indices]

    del x, y
    return x_train, y_train
x_train, y_train = preprocessPoliticalData('/kaggle/input/political-tweets/demfulltrain.txt', '/kaggle/input/political-tweets/repfulltrain.txt')
x_vtext, y_val = preprocessPoliticalData('/kaggle/input/political-tweets/repfullval.txt', '/kaggle/input/political-tweets/repfullval.txt')
from tensorflow.keras.layers import Activation, Dense, Dropout, Input, Embedding, GRU, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.models import Model

def biGRUnet():
    inputs = Input(name='inputs',shape=[x_train.shape[1]])
    layer = Embedding(10000,50,input_length=x_train.shape[1])(inputs)
    layer = Bidirectional(GRU(64, return_sequences = True), merge_mode='concat')(layer)
    layer = GlobalMaxPooling1D()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model
from tensorflow.keras.optimizers import RMSprop
print(x_train.shape[1])
with tpu_strategy.scope():
    model = biGRUnet()
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=16 * tpu_strategy.num_replicas_in_sync,epochs=20,
          validation_data=(x_vtext, y_val))
model.save("model.h5")
from transformers import TFBertForSequenceClassification

learning_rate = 2e-5
with tpu_strategy.scope():
    bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    bert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
bert_model.fit(x_train,y_train,batch_size=16 * tpu_strategy.num_replicas_in_sync,epochs=2, validation_data=(x_vtext, y_val))
bert_model.save("bert.h5")