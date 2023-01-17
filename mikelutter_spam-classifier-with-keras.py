import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
from IPython.display import display, HTML
import email
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
from nltk import FreqDist

spam_path = '../input/spam_2/spam_2/'
easy_ham_path = '../input/easy_ham/easy_ham/'
hard_ham_path = '../input/hard_ham/hard_ham/'
# label messagges according to folder
email_files = {'spam':     os.listdir(spam_path),
               'easy_ham': os.listdir(easy_ham_path),
               'hard_ham': os.listdir(hard_ham_path)
              }    
raw_data = []
labels = []
invalid_list = []
def processemail(body):
    body_pp = body.lower()
    body_pp = re.sub(r"<[^<>]+>", " html ", body_pp)
    body_pp = re.sub(r"[0-9]+", " number ", body_pp)
    body_pp = re.sub(r"(http|https)://[^\s]*", ' httpaddr ', body_pp)
    body_pp = re.sub(r"[^\s]+@[^\s]+", ' emailaddr ', body_pp)
    body_pp = re.sub(r"[$]+", ' dollar ', body_pp)
    body_pp = re.sub(r"[^a-zA-Z0-9]",' ', body_pp)
    return body_pp

def processfolder(path, label):
    for filename in os.listdir(path):
        #print(filename)
        try:
            file = open(path + filename,'r',errors='ignore')
            content = file.read()

            msg = email.message_from_string(content)
            if msg.is_multipart():
                body = []
                for payload in msg.get_payload():
                    # if payload.is_multipart(): ...
                    body.append(payload.get_payload())
                body = ' '.join(body)

            else:
                body = msg.get_payload()
            body = processemail(body)
            raw_data.append(body)
            labels.append(label)
        except:
          invalid_list.append(filename)
processfolder(spam_path, 1)
processfolder(easy_ham_path,0)
processfolder(hard_ham_path,0)
print("Total email count:{}".format(len(raw_data)))
print("Total labels: {}".format(len(labels)))
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

#train / test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(raw_data, labels, shuffle=True, test_size=0.33, random_state=42)

#tokenizing
tokenizer = keras.preprocessing.text.Tokenizer(num_words=4096)
tokenizer.fit_on_texts(X_train_raw)

#convert the words to token sequences
X_train = tokenizer.texts_to_sequences(X_train_raw)
X_test = tokenizer.texts_to_sequences(X_test_raw)

#pad the sequences
X_train = keras.preprocessing.sequence.pad_sequences(X_train, value=0, padding='post', maxlen=2048)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, value=0, padding='post', maxlen=2048)

print("Train size:{}".format(len(X_train)))
print("Test size:{}".format(len(X_test)))
#modeling
model = keras.Sequential()
model.add(keras.layers.Embedding(4096, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

print(model.summary())

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

#get a validation set
x_val = X_train[:788]
partial_x_train = X_train[788:]

y_val = y_train[:788]
partial_y_train = y_train[788:]
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=300,
                    batch_size=100,
                    validation_data=(x_val, y_val),
                    verbose=1, callbacks=[early_stop])
results = model.evaluate(X_test, y_test)
print("Final Test Set Results: {}".format(results))