# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import the required libraries

import keras

from keras.layers import Embedding

from keras.layers import Dense, Flatten, LSTM

from keras.layers import Input, GlobalMaxPool1D, Dropout

from keras.layers import Activation

from keras.models import Model, Sequential

from keras import optimizers



import os

import pandas as pd
#read the data files

train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
#lets look at the data

train.head()
#let's look at the test data

test.head()
#check for any null values in our dataset

print(train.isnull().any())
#again checking for any null values in the test dataset

print(test.isnull().any())
#divide our training data into features X and label Y

X_train = train['comment_text'] #will be used to train our model on

X_test = test['comment_text'] #will be used to predict the output labels to see how well our model has trained

y_train = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
y_train.shape #as expected total 6 columns for 6 predictor classes
#import the tokenizer class from the keras api

from keras.preprocessing.text import Tokenizer
#let's calculate the vocabulary size as well which will be given as an input to the Embedding layer

tokens = Tokenizer() #tokenizes our data

tokens.fit_on_texts(X_train)

vocab_size = len(tokens.word_index) + 1 #size of the total number of uniques tokens in our dataset

tokenized_train = tokens.texts_to_sequences(X_train) #converting our tokens into sequence of integers

tokenized_test = tokens.texts_to_sequences(X_test)
print(X_train[0]) #the first text

print(100 * '-')

print(tokenized_train[0]) #the correspondin first comment in the vectorized form
#import the pad_sequences class from the keras api

from keras.preprocessing.sequence import pad_sequences
max_len = 300 #maximum length of the padded sequence that we want (one of the hyperparameter that can be tuned)

padded_train = pad_sequences(tokenized_train, maxlen = max_len, padding = 'post') #post padding our sequences with zeros

padded_test = pad_sequences(tokenized_test, maxlen = max_len)
padded_train[:10] #as you can observe once our sentence ends the padding starts and continues until we have a vector of max_len
input_model = Input(shape = (max_len, ))

x = Embedding(input_dim = vocab_size, output_dim = 120)(input_model)

x = LSTM(60, return_sequences = True)(x)

x = Activation('relu')(x)

x = GlobalMaxPool1D()(x)

x = Dropout(0.2)(x)



# x = LSTM(100, return_sequences = True)(x)

# x = Activation('relu')(x)

# x = GlobalMaxPool1D()(x)

# x = Dropout(0.2)(x)



x = Dense(150, activation = 'relu')(x)

x = Dropout(0.5)(x)

x = Dense(100, activation = 'relu')(x)

x = Dropout(0.5)(x)

x = Dense(6, activation = 'softmax')(x)
model = Model(inputs = input_model, outputs = x)

optim = optimizers.Adam(lr = 0.01, decay = 0.01 / 64) #defining our optimizer

model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy']) #compiling the model
model.summary()
#fit the model onto our dataset

history = model.fit(padded_train, y_train, batch_size = 64, epochs = 5, validation_split = 0.3)
import numpy as np



embedding_dim = 50

#def create_embedding_matrix(filepath, embedding_dim):

vocab_size = len(tokens.word_index) + 1  # Adding again 1 because of reserved 0 index

embedding_matrix = np.zeros((vocab_size, embedding_dim))



with open('../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt', encoding = 'utf-8') as f:

    for line in f:

        word, *vector = line.split()

        if word in tokens.word_index:

            idx = tokens.word_index[word] 

            embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
model1 = Sequential()

model1.add(Embedding(vocab_size, embedding_dim, weights = [embedding_matrix], input_length = max_len, trainable = False))

model1.add(LSTM(60, return_sequences = True))

model1.add(Activation('relu'))

model1.add(GlobalMaxPool1D())

model1.add(Dropout(0.2))

model1.add(Dense(16, activation = 'relu'))

model1.add(Dropout(0.5))

model1.add(Dense(6, activation = 'softmax'))

model1.summary()
#compile the model

model1.compile(optimizer = optimizers.Adam(lr = 0.01, decay = 0.01 / 32), loss = 'categorical_crossentropy', metrics = ['accuracy'])
#fit the model on the dataset

history1 = model1.fit(padded_train, y_train, epochs = 5, batch_size = 32, validation_split = 0.3)
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]



y_pred = model1.predict(padded_test)



sample_submission = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv")



sample_submission[list_classes] = y_pred



sample_submission.to_csv("model_submission.csv", index = False)