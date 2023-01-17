import re
import pickle
import os
import logging
import json
import pickle
from gensim import corpora
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
DATA_DIR = '../input/vntcdata'
DATA_DIR_TRAIN = os.path.join(DATA_DIR, 'train_dict.json')
DATA_DIR_TEST = os.path.join(DATA_DIR, 'test_dict.json')

def read_txt(file_path):
    with open(file_path, 'r', encoding = 'utf-16-le') as f:
        s = f.read().lower()
        s = re.sub('\ufeff+|\n+', ' ', s)
    return s

def read_json(file_path):
    with open(file_path, 'r', encoding = 'utf-16-le') as f:
        data = json.load(f)
    return data

def save_json(obj, file_name):
    with open(file_name, 'w', encoding = 'utf-16-le') as f:
        json.dump(obj, f)
        
def read_stopwords(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
    return stopwords

def save_pickle(obj, file_path):
    with open(file_path, 'wb') as outfile:
        pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)
        
def load_pickle(file_path):
    with open(file_path, 'rb') as outfile:
        obj = pickle.load(outfile)
    return obj

TOPIC_LIST = [list(read_json('../input/10topicvntc/topic.json'))]
topic_dict = corpora.Dictionary(TOPIC_LIST)
TOPIC = topic_dict.token2id
TOPIC
def read_data(folder_path):
    topics = os.listdir(folder_path)
    data = {'file_names':[], 'topic_ids':[],'contents': []}
    for topic in topics:
        folder_path_topic = os.path.join(folder_path, topic)
        file_names = os.listdir(folder_path_topic)
        data['file_names'] += file_names
        data['topic_ids'] += [TOPIC[topic]]*len(file_names)
        contents = [read_txt(os.path.join(folder_path_topic, file_name)) for file_name in file_names]
        data['contents'] += contents
    return data

# train_dict = read_data(DATA_DIR_TRAIN)
# save_json(train_dict, 'data/train.json')
train_dict = read_json(DATA_DIR_TRAIN)
# test_dict = read_data(DATA_DIR_TEST)
# save_json(test_dict, 'data/test.json')
# test_dict = read_json(DATA_DIR_TEST)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
%matplotlib inline

# Kiểm tra phân phối số lượng bài báo của mỗi topic_ids
sns.countplot(train_dict['topic_ids'])
plt.xlabel('Label')
plt.title('Number of each topics')
X = train_dict['contents']
y = train_dict['topic_ids']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, stratify = y)
sns.countplot(y_train)
plt.xlabel('Label')
plt.title('Number of each topics')
sns.countplot(y_test)
plt.xlabel('Label')
plt.title('Number of each topics')
max_words = 3000
max_len = 150
tok = Tokenizer(num_words = max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen = max_len)
print('sequences[2] length: ', len(sequences[2]))
print('sequences length: ', len(sequences))
print('sequences_matrix shape: ', sequences_matrix.shape)
print('sequences_matrix first row: ', sequences_matrix[1, :])
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    # Embedding (input_dim: size of vocabolary, 
    # output_dim: dimension of dense embedding, 
    # input_length: length of input sequence)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dense(128,name='FC2')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(10,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model
model = RNN()
model.summary()
model.compile(loss = 'sparse_categorical_crossentropy', \
              optimizer = RMSprop(), metrics = ['accuracy'])
model.fit(sequences_matrix, y_train, batch_size = 10, epochs = 10,
          validation_split = 0.2, callbacks = \
          [EarlyStopping(monitor = 'val_loss', min_delta = 0.01)])
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
accr = model.evaluate(test_sequences_matrix,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
y_pred = model.predict_classes(test_sequences_matrix)
print(classification_report(y_pred, y_test))