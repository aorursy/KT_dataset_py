# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import pandas as pd

filename = "../input/nlp-disaster/train.csv"

data = pd.read_csv(filename)

data.head()





import nltk

from nltk.corpus import stopwords



from numpy import array

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense

from keras.layers import Flatten

from keras.layers import GlobalMaxPooling1D

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer





test = pd.read_csv('../input/nlp-disaster/test.csv')

y_test = pd.read_csv('../input/nlp-disaster/sample_submission.csv')

test
import pandas as pd 

import numpy as np

y_test = pd.Series(y_test["target"])

y_test
table = data.groupby([ "target"]).size()

table
data["text"].head()


data = data.filter(["target", "text"])

data
y = data['target']

X = data['text']
from sklearn.model_selection import train_test_split

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords





#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train = pd.Series(data['text'])

y_train = pd.Series(data['target'])

X_test = pd.Series(test['text'])

#y_test = pd.Series(test['target'])

type(X_train)
tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(X_train)



X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)

X_test
vocab_size = len(tokenizer.word_index) + 1



maxlen = 100



X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

from numpy import array

from numpy import asarray

from numpy import zeros



embeddings_dictionary = dict()

glove_file = open('../input/eword-embedding-glove/glove.6B.100d.txt')



for line in glove_file:

    records = line.split()

    word = records[0]

    vector_dimensions = asarray(records[1:], dtype='float32')

    embeddings_dictionary [word] = vector_dimensions

glove_file.close()





embedding_matrix = zeros((vocab_size, 100))

for word, index in tokenizer.word_index.items():

    embedding_vector = embeddings_dictionary.get(word)

    if embedding_vector is not None:

        embedding_matrix[index] = embedding_vector
# Importing the Keras libraries and packages

import tensorflow as tf

from tensorflow import keras

from keras import layers

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense

from keras.layers import Flatten

from keras.layers import GlobalMaxPooling1D

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer





# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim = 100))



# Adding the second hidden layer

classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 100, epochs = 100, verbose=1, validation_split=0.2)



#y_test1 = y_test.filter(['target'], axis = 1)

#y_test2 =  y_test1.to_numpy()

y_pred = classifier.predict(X_test)



y_pred1 = (y_pred > 0.5)

y_pred1
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred1)

cm
y_pred_1 = pd.DataFrame(y_pred,columns = ['Prediction'])

a = round(y_pred_1, 2)

y_test = pd.read_csv('../input/nlp-disaster/sample_submission.csv')



new = pd.concat([y_test,y_pred_1], axis= 1)

new["Predict"]= 1

new["Predict"][new['Prediction'] < 0.5] =  0

#new = new.drop(['Prediction', 'id'], axis = 1)

new = new.drop(["target", "Prediction"], axis = 1)

new.columns = ["id", "target"]

new
y_pred_1 = pd.DataFrame(y_pred,columns = ['Prediction'])

a = round(y_pred_1, 2)

y_test = pd.read_csv('../input/nlp-disaster/sample_submission.csv')



new = pd.concat([y_test,y_pred_1], axis= 1)

new["Predict"]= 1

new["Predict"][new['Prediction'] < 0.5] =  0

#new = new.drop(['Prediction', 'id'], axis = 1)

new = new.drop(["target", "Prediction"], axis = 1)

new.columns = ["id", "target"]

new