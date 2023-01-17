#Importing necessary modules

import numpy as np 

import pandas as pd 

import os

import seaborn as sns

import matplotlib.pyplot as plt

import string

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.models import Model

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Conv1D

from keras.optimizers import Adam

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping

from nltk.corpus import stopwords
#Load kaggle dataset into dataframe

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/trip-advisor-hotel-reviews/tripadvisor_hotel_reviews.csv')
#Examine first few rows of dataframe

df.head()
#Label count for each label

df['Rating'].value_counts()
#Visualzing the label count

sns.countplot(df.Rating)
#converting review text to lower case



dataset = df['Review'].str.lower()
#removing the punctuation marks



PUNCTUATION = string.punctuation

def remove_punctuation(text):

    return text.translate(str.maketrans('', '', PUNCTUATION))



df["Review"] = df["Review"].apply(lambda sentence: remove_punctuation(sentence))

df.head()
#removing the words whose length is less than 2 and also numbers if present

def remove_stopwords(text):

    return " ".join([word for word in str(text).split() if ((len(word)>2) and (word.isalpha()))])



df["Review"] = df["Review"].apply(lambda sentence: remove_stopwords(sentence))

df.head()
#stop words list

stop_words_list = stopwords.words('english')
#removing the stop words from the review text

def remove_stopwords(text):

    return " ".join([word for word in str(text).split() if word not in stop_words_list])



df["Review"] = df["Review"].apply(lambda sentence: remove_stopwords(sentence))

df.head()
#removing blank characters from the review text

def remove_blankCharacters(text):

    return " ".join([word for word in str(text).split() if word not in ['',' ']])



df["Review"] = df["Review"].apply(lambda sentence: remove_blankCharacters(sentence))

df.head()
#Calculating average length of the sentences

def calculate_length(text):

    words = [word for word in str(text).split()]

    return len(words)



df["length"] = df["Review"].apply(lambda sentence: calculate_length(sentence))

avg_len = df["length"].mean()

df.head()

print("Average length of the sentences is {}".format(avg_len))
#Calculating maximum and minimum length of review sentences

maximum = df['length'].max()

minimum = df['length'].min()



print('maximum length is {}, minimum length is {}'.format(maximum,minimum))

df.head()
#Preparing the labels and the train data

le = LabelEncoder()

Y = le.fit_transform(df.Rating)

Y = Y.reshape(-1,1)

X = df['Review'].tolist()
#splitting the data into train and test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0, stratify=Y)
print('No. of train samples: {}'.format(len(X_train)))

print('No. of train labels: {}\n'.format(len(Y_train)))

print('No. of test samples: {}'.format(len(X_test)))

print('No. of test labels: {}'.format(len(Y_test)))
#convert train labels into one hot vectors

Y_train_one_hot_labels = to_categorical(Y_train)
#Tokenizing, converting to sequences and padding the train data 

max_words = 3000

max_len = 200

tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(X_train)

sequences = tokenizer.texts_to_sequences(X_train)

sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
#model

inputs = Input(name='inputs',shape=[max_len])

layer = Embedding(max_words,100,embeddings_initializer="uniform",input_length=max_len)(inputs)

layer = Conv1D(128, 5, activation='relu')(layer)

layer = LSTM(128)(layer)

layer = Dropout(0.2)(layer)

layer = Dense(512, name='FC1', activation='relu')(layer)

layer = Dropout(0.2)(layer)

layer = Dense(128, name='FC2')(layer)

layer = Dropout(0.2)(layer)

layer = Dense(64, name='FC3')(layer)

layer = Dense(5, name='out_layer')(layer)

layer = Activation('softmax')(layer)

model = Model(inputs=inputs,outputs=layer)

model.summary()

model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
#Training



model.fit(sequences_matrix,Y_train_one_hot_labels,batch_size=128,epochs=5,

          validation_split=0.1)
#convert test data into sequences and padding them

test_sequences = tokenizer.texts_to_sequences(X_test)

test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
#Calculating the accuracy of the model

Y_test_one_hot_labels = to_categorical(Y_test)

accr = model.evaluate(test_sequences_matrix,Y_test_one_hot_labels)
print('Loss: {:0.3f}  Accuracy: {:0.3f}'.format(accr[0],accr[1]))