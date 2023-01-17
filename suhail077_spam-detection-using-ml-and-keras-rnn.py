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
data = pd.read_csv('../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')

data.head()
data['Category'] = [1 if cat == "spam" else 0 for cat in data["Category"]]

data.head()
import nltk as nlp

import re

message_list = []

for message in data["Message"]:

    # Remove everything other than alphabets

    message = re.sub("[^a-zA-Z]"," ",message)

    # Characters to lower case

    message = message.lower()   

    # Tokenize to list

    message = nlp.word_tokenize(message)

    # Find Lemma 

    lemma = nlp.WordNetLemmatizer()

    message = [ lemma.lemmatize(word) for word in message]

    # Convert back to string

    message = " ".join(message)

    message_list.append(message) #we hide all word one section

message_list
from sklearn.feature_extraction.text import CountVectorizer 

max_features = 3000

count_vectorizer = CountVectorizer(max_features = max_features, stop_words = "english")

sparce_matrix_features = count_vectorizer.fit_transform(message_list).toarray()

print("the most using {} words: {}".format(max_features,count_vectorizer.get_feature_names()))
# Categories List

y = data.iloc[:,0].values

# Features List

x= sparce_matrix

# Train and Test Split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 111)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("Accuracy: ",format(nb.score(x_test,y_test)))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter = 200)

lr.fit(x_train,y_train)

print('Accuracy: ', format(lr.score(x_test,y_test)))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

print('With KNN (K=3) Accuracy: ',knn.score(x_test,y_test))
print(x_train.shape)

print(x_test.shape)

x_train = x_train.reshape(4457,3000,1)

x_test = x_test.reshape(1115,3000,1)



from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout



# Initialising the RNN

regressor = Sequential()



# Adding the first LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))

regressor.add(Dropout(0.2))



# Adding a second LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))



# Adding a third LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))



# Adding a fourth LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50))

regressor.add(Dropout(0.2))



# Adding the output layer

regressor.add(Dense(units = 1))



# Compiling the RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=["accuracy"])



# Fitting the RNN to the Training set

regressor.fit(x_test, y_test, epochs = 3, batch_size = 32)
