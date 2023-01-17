"""Importing the required libraries"""



import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import load_model

from keras.layers import *

from keras import backend, Model
"""Reading the training dataset and performing the EDA ( Exploratory Data Analysis ) in the upcoming cells"""



path = '../input/'

comp = 'jigsaw-toxic-comment-classification-challenge/'

dataset = pd.read_csv('../input/train.csv')
"""Performing EDA ( Exploratory Data Analysis ) """





#Having a look at the data

print("Number of rows in data = ", dataset.shape[0])

print("Number of columns in data = ", dataset.shape[1])

print("\n")

print("---Sample data---")

dataset.head()
"""Performing EDA ( Exploratory Data Analysis ) """



#Visualising the Training data : Number of comments vs. Label

labels = dataset.columns.values[2:]

labels_count = dataset.iloc[:, 2:].sum().values

sns.set(font_scale = 2)

plt.figure(figsize = (15, 8))

fig = sns.barplot(labels, labels_count)

plt.title("Comments vs. Label", fontsize = 24)

plt.ylabel('Number of comments', fontsize = 20)

plt.xlabel('Label ', fontsize = 20)

rects = fig.patches

for rect, label in zip(rects, labels_count):

    height = rect.get_height()

    fig.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha = 'center', va = 'bottom', fontsize = 18)

plt.show()
"""Function for Text Preprocessing"""



#Removing non-english symbols, HTML tags, converting to lower-case, lemmatizing, and finally removing the stop-words 

stop_words = set(stopwords.words("english")) 

stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])

lemmatizer = WordNetLemmatizer()



def clean_text(X):

    processed = []

    for text in X:

        text = text[0]

        text = re.sub(r'[^\w\s]', '',text, re.UNICODE)

        text = re.sub('\n', ' ',text, re.UNICODE)

        text = re.sub('<.*?>', '', text)

        text = text.lower()

        text = [lemmatizer.lemmatize(token) for token in text.split(" ")]

        text = [lemmatizer.lemmatize(token, "v") for token in text]

        text = [word for word in text if not word in stop_words]

        processed.append(text)

    return processed
"""Getting the X and Y and preprocessing them"""



X = dataset.iloc[:, 1:2].values

y_train = dataset.iloc[:, 2:8].values

X_train = clean_text(X)
"""Tokenization and Padding"""

vocab_size = 10000

maxlen = 250

embed_dim = 20

batch_size = 64

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train)

tokenized_word_list = tokenizer.texts_to_sequences(X_train)

X_train_padded = pad_sequences(tokenized_word_list, maxlen = maxlen, padding='post')
"""EarlyStopping and ModelCheckpoint"""



es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 2)

mc = ModelCheckpoint('model_best.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)
"""Creating TextCNN model for comment classification"""



#Defining the Layers of the model

input_X = Input(shape=(maxlen, ))

embed = Embedding(vocab_size, embed_dim)(input_X)

conv_1 = Conv1D(filters = 32, kernel_size = 2, activation = 'relu', padding = 'valid')(embed)

out_1 = GlobalMaxPooling1D()(conv_1)

conv_2 = Conv1D(filters = 32, kernel_size = 4, activation = 'relu', padding = 'valid')(embed)

out_2 = GlobalMaxPooling1D()(conv_2)

conc = concatenate([out_1, out_2])

dense1 = Dense(32, activation = 'relu')(conc)

out = Dense(6, activation = 'sigmoid', name = 'output_layer')(dense1)



#Defining the model now

model = Model(input_X, out)

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.summary()
"""Fitting the model"""

X_train, X_val, y_train, y_val = train_test_split(X_train_padded, y_train, test_size = 0.2, random_state = 42, shuffle = True)

model.fit(X_train, y_train, epochs = 10, batch_size = batch_size, verbose = 1, validation_data = [X_val, y_val], callbacks = [es, mc])
"""Importing the Test Data and making it ready to be passed to the Model"""



dataset2 = pd.read_csv('../input/test.csv')

X_test = dataset2.iloc[:, 1:2].values

X_test = clean_text(X_test)

tokenized_word_list = tokenizer.texts_to_sequences(X_test)

X_test_padded = pad_sequences(tokenized_word_list, maxlen = maxlen, padding='post')
"""Testing and creating the test results"""



labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

model = load_model('model_best.h5')

y_test = model.predict(X_test_padded, batch_size = 512, verbose = 1)

sample_submission = pd.read_csv(f'{path}sample_submission.csv')

sample_submission[labels] = y_test

sample_submission.to_csv('submission.csv', index = False)