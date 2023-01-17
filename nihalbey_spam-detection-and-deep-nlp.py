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
#Firstly we read our data

data = pd.read_csv("../input/SPAM text message 20170820 - Data.csv")
data.head()

#We must change category 1 or 0

data["Category"] = [1 if each == "spam" else 0 for each in data["Category"]]
data.head()
#We choose 1 row.And we throw punctuation

import re

nlp_data = str(data.iloc[2,:])

nlp_data = re.sub("[^a-zA-Z]"," ",nlp_data)
#After return lower case

nlp_data = nlp_data.lower()
#we have two choice we can use split methot or tokenize

import nltk as nlp

nlp_data = nlp.word_tokenize(nlp_data)

#nlp_data = nlp_data.split() or we can do so
#we have to find word root

lemma = nlp.WordNetLemmatizer()

nlp_data = [lemma.lemmatize(word) for word in nlp_data]
#We join our data

nlp_data = " ".join(nlp_data)
import nltk as nlp

import re

description_list = []

for description in data["Message"]:

    description = re.sub("[^a-zA-Z]"," ",description)

    description = description.lower()   # buyuk harftan kucuk harfe cevirme

    description = nlp.word_tokenize(description)

    #description = [ word for word in description if not word in set(stopwords.words("english"))]

    lemma = nlp.WordNetLemmatizer()

    description = [ lemma.lemmatize(word) for word in description]

    description = " ".join(description)

    description_list.append(description) #we hide all word one section
#We make bag of word it is including number of all word's info

from sklearn.feature_extraction.text import CountVectorizer 

max_features = 3000 #We use the most common word

count_vectorizer = CountVectorizer(max_features = max_features, stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()

print("the most using {} words: {}".format(max_features,count_vectorizer.get_feature_names()))
#We separate our data is train and test

y = data.iloc[:,0].values   # male or female classes

x = sparce_matrix

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)
#We make model for predict

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("the accuracy of our model: {}".format(nb.score(x_test,y_test)))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter = 200)

lr.fit(x_train,y_train)

print("our accuracy is: {}".format(lr.score(x_test,y_test)))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

#print('Prediction: {}'.format(prediction))

print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test))
x_test = x_test.reshape(558,3000,1)

x_train = x_train.reshape(5014,3000,1)
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
