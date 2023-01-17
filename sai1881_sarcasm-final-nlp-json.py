# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input, Flatten
from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
#
# Download glove from Google Drive
#



import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f: 
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                

file_id = '1d8Yb9PkLN-HdS0_D9AfL5e5aaiyhhwxc'
destination = './glove.6B.50d.txt'
download_file_from_google_drive(file_id, destination)
#
# Load data
#

data_json = []
import json
for line in open('/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json', 'r'):
    data_json.append(json.loads(line))
# To pandas
data = pd.DataFrame(data_json)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)
# Check for imbalance
Counter(data.is_sarcastic)
import nltk
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
is_noun = lambda pos: pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'
debug = True

# Cleaning based on POS
# Also tried Spelling correction

def transform(text):
    tokenized = nltk.word_tokenize(text.encode('ascii', 'ignore').decode("utf-8") )
    description = []
    
    
    for (word, pos) in nltk.pos_tag(tokenized):

        if is_noun(pos):
            word = lemmatizer.lemmatize(word, "n")
            description.append(word)
        else:
            word = lemmatizer.lemmatize(word, "v")
            
            # Spelling correction
#             corrected = spell.correction(word)
            
            
#             if corrected != word and debug:
#                 print("corrected", corrected, "original", word, "pos", pos)
                
                
            description.append(word)
    return " ".join(description)
        
text = "headline"
target = "is_sarcastic"
data[text] = data.headline.apply(transform)
# Based glove comparison, Doing some replacements to combine similar words
punctuations = '@#!?+&*[]-%.:/();$=><|{}^”“' + "`"

for p in punctuations:
    data[text] = data[text].str.replace(p, f' {p} ') 
    
data[text] = data[text].str.replace(r"’ll", ' will')
data[text] = data[text].str.replace(r"’hv", ' have')
data[text] = data[text].str.replace(r"'", '')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data[text], data[target].values , test_size=0.2)
max_features = 2000
max_length = 1000

# Tokenizer

tokenizerx = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
tokenizerx.fit_on_texts(x_train)
x_train = tokenizerx.texts_to_sequences(x_train) 
x_train = pad_sequences(x_train,maxlen = max_length, padding = 'post').astype(float)
x_test = tokenizerx.texts_to_sequences(x_test) 
x_test = pad_sequences(x_test,maxlen = max_length, padding = 'post').astype(float)
#
# Checking for non existing words by comparing with glove
#

import csv
words = pd.read_table("./glove.6B.50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

all_words = list(tokenizerx.word_docs)

exist = []
not_exist = []

for word in all_words:
    if word in  words.index:
        exist.append(word)
    else:
        not_exist.append(word)
        
len(all_words), len(exist), len(not_exist)
x_train = np.expand_dims(x_train, axis = 2)
x_test = np.expand_dims(x_test, axis = 2)
x_train = x_train.astype(float)
x_test = x_test.astype(float)

y_train = y_train.astype(float)
y_test = y_test.astype(float)
metrics = [keras.metrics.TruePositives(name='tp'),keras.metrics.FalsePositives(name='fp'),keras.metrics.TrueNegatives(name='tn'),keras.metrics.FalseNegatives(name='fn'),keras.metrics.BinaryAccuracy(name='accuracy'),keras.metrics.Precision(name='precision'),keras.metrics.Recall(name='recall'),keras.metrics.AUC(name='auc')]  
# define the model
FILTERS = 3
POOL = 3

kernal_size = 20
model = keras.Sequential()
model.add(keras.layers.Embedding(max_features, 24, input_length=max_length))




# model.add(Conv1D(8, FILTERS, activation="relu", padding='same'))
# model.add(MaxPooling1D(POOL))
# model.add(BatchNormalization())
# model.add(Conv1D(16, FILTERS, activation="relu", padding='same'))
# model.add(MaxPooling1D(POOL))
# model.add(BatchNormalization())

model.add(keras.layers.LSTM(40, return_sequences=True))
model.add(keras.layers.Dropout(0.15))
model.add(keras.layers.LSTM(40))
model.add(keras.layers.Dropout(0.15))

model.add(keras.layers.Flatten())

model.add(Dense(100, kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0.001)))

model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=keras.losses.BinaryCrossentropy(), metrics=[metrics])
print(model.summary())

keras.backend.set_value(model.optimizer.lr,0.0001)
class_weight = dict(Counter(y_test))
total = len(y_test)
print(dict(Counter(y_test)))
class_weight = {i:(1/j)*total/len(class_weight) for i,j in class_weight.items()}
# class balancing for TF
class_weight
model.fit(x_train, y_train , validation_data=(x_test, y_test), batch_size=100, epochs=10, verbose=1, class_weight=class_weight)
