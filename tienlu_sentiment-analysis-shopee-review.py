import tensorflow as tf
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print(os.listdir('../input/shopee-sentiment-analysis'))
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from bs4 import BeautifulSoup
import re

from tqdm import tqdm
from keras.utils import to_categorical
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense,Dropout,Embedding,LSTM
from keras.callbacks import EarlyStopping

from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Sequential

tf.random.set_seed(123)
train = pd.read_csv('../input/shopee-sentiment-analysis/train.csv')
test = pd.read_csv('../input/shopee-sentiment-analysis/test.csv')

print(train.head())
train.shape
print(test.head())
test.shape
def clean_sentences(df):
    reviews = []

    for sent in tqdm(df['review']):        
        review_text = BeautifulSoup(sent).get_text()        
        review_text = re.sub("[^a-zA-Z]"," ", review_text)    
        words = word_tokenize(review_text.lower())    
        lemma_words = [lemmatizer.lemmatize(i) for i in words]    
        reviews.append(lemma_words)
    return(reviews)
train_sentences = clean_sentences(train)
test_sentences = clean_sentences(test)
print(len(train_sentences))
print(len(test_sentences))
print(type(train_sentences))
print(len(train_sentences))
train['rating'] = train['rating'] -1
target = train['rating'].values 
y_target = to_categorical(target)
num_classes = y_target.shape[1]

print(target)
print(y_target)
print(num_classes)
print(len(y_target))
X_train,X_val,y_train,y_val = train_test_split(train_sentences, y_target, test_size = 0.2, stratify = y_target)
print(len(X_train))
print(len(X_val))
print(len(y_train))
print(len(y_val))
unique_words = set()
len_max = 0

for sent in tqdm(X_train):
    unique_words.update(sent)
    if(len_max < len(sent)):
        len_max = len(sent)
        
print(len(list(unique_words)))
print(len_max)
tokenizer = Tokenizer(num_words = len(list(unique_words)))
tokenizer.fit_on_texts(list(X_train))

X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(test_sentences)

X_train = sequence.pad_sequences(X_train, maxlen = len_max)
X_val = sequence.pad_sequences(X_val, maxlen = len_max)
X_test = sequence.pad_sequences(X_test, maxlen = len_max)

print(X_train.shape, X_val.shape, X_test.shape)
early_stopping = EarlyStopping(min_delta = 0.001, mode = 'max', monitor = 'val_accuracy', patience = 2)
callback = [early_stopping]
model = Sequential()
model.add(Embedding(len(list(unique_words)), 300, input_length = len_max))
model.add(LSTM(128, dropout = 0.5, recurrent_dropout = 0.5, return_sequences = True))
model.add(LSTM(64, dropout = 0.5, recurrent_dropout = 0.5, return_sequences = False))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 0.005), metrics = ['accuracy'])
model.summary()
history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 6, batch_size = 256, verbose = 1, callbacks = callback)
import matplotlib.pyplot as plt

# Create count of the number of epochs
epoch_count = range(1, len(history.history['loss']) + 1)

# Visualize learning curve. Here learning curve is not ideal. It should be much smoother as it decreases.
#As mentioned before, altering different hyper parameters especially learning rate can have a positive impact
#on accuracy and learning curve.
plt.plot(epoch_count, history.history['loss'], 'r--')
plt.plot(epoch_count, history.history['val_loss'], 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

y_pred=model.predict_classes(X_test)

print(len(y_pred))
sub_file = pd.read_csv('../input/shopee-sentiment-analysis/test.csv')
print(sub_file.shape)
sub_file = sub_file[['review_id']]
sub_file['rating']= y_pred
sub_file['rating'] = sub_file['rating'] + 1
sub_file.to_csv('Submission.csv',index=False)