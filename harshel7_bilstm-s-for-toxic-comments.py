# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



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

from keras.layers import Bidirectional

from keras.layers import BatchNormalization

from keras.models import Model, Sequential

from keras import optimizers



import os

import pandas as pd
#read the data files

train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
#divide our training data into features X and label Y

X_train = train['comment_text'] #will be used to train our model on

X_test = test['comment_text'] #will be used to predict the output labels to see how well our model has trained

y_train = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
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

            embedding_matrix[idx] = np.array(vector, dtype = np.float32)[:embedding_dim]
model = Sequential()

model.add(Embedding(vocab_size, embedding_dim, weights = [embedding_matrix], input_length = max_len, trainable = False))

model.add(Bidirectional(LSTM(50, return_sequences = True)))

model.add(GlobalMaxPool1D())

model.add(BatchNormalization())

model.add(Dropout(0.1))

model.add(Dense(50, activation = "relu"))

model.add(Dropout(0.1))

model.add(Dense(32, activation = "relu"))

model.add(Dropout(0.1))

model.add(Dense(6, activation = 'sigmoid'))

model.summary()
#compile the model

model.compile(loss = 'binary_crossentropy', optimizer = optimizers.Adam(lr = 0.01, decay = 0.01/32), metrics = ['accuracy'])
#fit the model on the dataset

history = model.fit(padded_train, y_train, epochs = 5, batch_size = 128, validation_split = 0.3)
#list all the output class labels

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]



#make the predictions

y_pred = model.predict(padded_test, verbose = 1, batch_size = 128)
#making submission

#read in the submission file

sample_submission = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv")



sample_submission[list_classes] = y_pred



sample_submission.to_csv("BiLSTM_submission.csv", index = False)