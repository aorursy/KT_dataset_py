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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Input, Conv1D, BatchNormalization
from tensorflow.keras import utils
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

mb_df = pd.read_csv('/kaggle/input/mbti-type/mbti_1.csv')
mb_df.head()
# Source: https://www.kaggle.com/anasofiauzsoy/myers-briggs-types-with-tensorflow-bert

import string
import re

def clean_text(text):
    regex = re.compile('[%s]' % re.escape('|'))
    text = regex.sub(" ", text)
    words = str(text).split()
    words = [i.lower() + " " for i in words]
    words = [i for i in words if not "http" in i]
    words = " ".join(words)
    words = words.translate(words.maketrans('', '', string.punctuation))
    return words
mb_df['clean_text'] = mb_df['posts'].apply(clean_text)
mb_df['type_factorized'], names = pd.factorize(mb_df['type'])
mb_df.head()
X = mb_df['clean_text']
y = mb_df['type_factorized']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=22)
tfvect = TfidfVectorizer(max_features=500)
X_train_vect = tfvect.fit_transform(X_train)
X_test_vect = tfvect.transform(X_test)
X_train_vect = X_train_vect.toarray()
X_test_vect = X_test_vect.toarray()
y_train_cat = utils.to_categorical(y_train, num_classes=16)
y_test_cat = utils.to_categorical(y_test, num_classes=16)
X_train_vect.shape
X_test_vect.shape
model = Sequential()

model.add(Input(shape=(500, )))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(16, activation = 'softmax'))
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', 'Recall'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(X_train_vect,
                    y_train_cat,
                    validation_data=(X_test_vect, y_test_cat),
                    callbacks = callback,
                    epochs=30)
train_loss = history.history['loss']
test_loss = history.history['val_loss']
epoch_labels = history.epoch

# Set figure size.
plt.figure(figsize=(12, 8))

# Generate line plot of training, testing loss over epochs.
plt.plot(train_loss, label='Training Loss', color='#185fad')
plt.plot(test_loss, label='Testing Loss', color='orange')

# Set title
plt.title('Training and Testing Loss by Epoch', fontsize=25)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Categorical Crossentropy', fontsize=18)
plt.xticks(epoch_labels, epoch_labels)    # ticks, labels

plt.legend(fontsize=18);
train_acc = history.history['accuracy']
test_acc = history.history['val_accuracy']
epoch_labels = history.epoch

# Set figure size.
plt.figure(figsize=(12, 8))

# Generate line plot of training, testing loss over epochs.
plt.plot(train_acc, label='Training Accuracy', color='#185fad')
plt.plot(test_acc, label='Testing Accuracy', color='orange')

# Set title
plt.title('Training and Testing Accuracy by Epoch', fontsize=25)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Categorical Crossentropy', fontsize=18)
plt.xticks(epoch_labels, epoch_labels)    # ticks, labels

plt.legend(fontsize=18);
train_rec = history.history['recall']
test_rec = history.history['val_recall']
epoch_labels = history.epoch

# Set figure size.
plt.figure(figsize=(12, 8))

# Generate line plot of training, testing loss over epochs.
plt.plot(train_rec, label='Training Recall', color='#185fad')
plt.plot(test_rec, label='Testing Recall', color='orange')

# Set title
plt.title('Training and Testing Recall by Epoch', fontsize=25)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Categorical Crossentropy', fontsize=18)
plt.xticks(epoch_labels, epoch_labels)    # ticks, labels

plt.legend(fontsize=18);
preds = model.predict_classes(X_test_vect)
preds
preds.shape
y_test.shape
tf.math.confusion_matrix(y_test, predictions=preds)
import tensorboard
import datetime
%reload_ext tensorboard
logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

history = model.fit(X_train_vect,
                    y_train_cat,
                    validation_data=(X_test_vect, y_test_cat),
                    callbacks = [tb_callback],
                    epochs=15)
%tensorboard --logdir logs/fit
