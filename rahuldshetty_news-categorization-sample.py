# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing the pandas library

import numpy as np

import pandas as pd



# loading the data

path = "/kaggle/input/bbc-fulltext-and-category/bbc-text.csv"

data = pd.read_csv(path)
# printing out few elements

data.head()
# Basic details about the dataset

data.describe()
classes = data['category'].unique()

print("Different Categories:",classes)
data['category'].value_counts()
# importing the matplotlib plots

import seaborn as sns



sns.countplot(x='category',data = data)
#importing libraries

import nltk

import re

from nltk.corpus import stopwords 

from nltk.stem.porter import PorterStemmer 

from nltk.stem import WordNetLemmatizer 
# loading the stop words list, stemmer and lemmatizer

stop_words = set(stopwords.words("english")) 

stemmer = PorterStemmer() 

lemmatizer = WordNetLemmatizer() 

print(stop_words)
# methods to perform preprocessing

def process_words(text):

    # tokenize the text

    words = text.split()

    new_words_list = []

    

    for word in words:

        # only add words which are not stop words

        if word not in stop_words:

            word = stemmer.stem(word)

            word = lemmatizer.lemmatize(word, pos ='v')

            new_words_list.append(word)

    

    # concatenate the string

    return " ".join(new_words_list)





def preprocess(text):

    # convert to lower case

    text = text.lower()

    

    # replace non-alphabets with null

    text = re.sub('[^a-zA-Z ]','',text)

    

    #remove stop words

    text = process_words(text)

    

    return text
sample = data['text'][50]



print("Sample Text before Pre-Processing:\n",sample)
pre_sample = preprocess(sample)

print("Sample Text after Pre-Processing:\n",pre_sample)
# apply preprocessing on entire dataset

x = data['text'].apply(lambda x:preprocess(x))

print(x[:1])
# convert the input text to suitable features

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



count_vectorizer = CountVectorizer()

tfidf_vectorizer = TfidfVectorizer()



x_counts = count_vectorizer.fit_transform(x)

x_tfidf = tfidf_vectorizer.fit_transform(x)
new_x = x_counts
# create labels for target

from sklearn.preprocessing import LabelEncoder,OneHotEncoder 





label_encoder = LabelEncoder()

y = label_encoder.fit_transform( data['category'] )

print("Label Encodings:",y)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(new_x,y,test_size=0.15)

print("Size of Training data:",x_train.shape[0])

print("Size of Testing data:",x_test.shape[0])
# Loading the libraires

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
nb_model = MultinomialNB()

logistic_model = LogisticRegression()

dec_tree_model = DecisionTreeClassifier()
# training the model

nb_model.fit(x_train,y_train)

logistic_model.fit(x_train,y_train)

dec_tree_model.fit(x_train,y_train)
def check_accuracy(model,x_test,y_test):

    total = x_test.shape[0]

    count =  0

    res = model.predict(x_test)

    for i in range(total):

        y_true = y_test[i]

        if y_true == res[i]:

            count+=1

    return count/total



print("Naive Bayes Accuracy:",check_accuracy(nb_model,x_test,y_test))

print("Logistic Regression Accuracy:",check_accuracy(logistic_model,x_test,y_test))

print("Desicion Tree Accuracy:",check_accuracy(dec_tree_model,x_test,y_test))

sample_data = "actor injured while shooting and the movie got cancelled"

processed_data = preprocess(sample_data)

x_sample = count_vectorizer.transform([processed_data])



# get results

nb_result =  nb_model.predict(x_sample)

log_result = logistic_model.predict(x_sample)

tree_result = dec_tree_model.predict(x_sample)



# get labels

nb_result = label_encoder.inverse_transform(nb_result)

log_result = label_encoder.inverse_transform(log_result)

tree_result = label_encoder.inverse_transform(tree_result)



print("Naive Bayes Result:",nb_result)

print("Logistic Regession Result:",log_result)

print("Decision Tree Result:",tree_result)
input_shape = x_train[0].shape[1]

output_shape = 5

print("Input Shape:",input_shape)

print("Output Shape:",output_shape)
#create one hot encoding's for the labels

onehotencoder = OneHotEncoder() 

onehotencoder.fit(y.reshape(-1, 1))

labels = onehotencoder.transform(y.reshape(-1, 1)).toarray()

print("One hot Vector:\n",labels)
x_train, x_test, y_train, y_test = train_test_split(new_x,labels,test_size=0.15)
# import library

from keras.models import Sequential

from keras.layers import Dense, Dropout
model = Sequential()

model.add(Dense( 1024 , activation='sigmoid',  input_dim = input_shape  ))

model.add(Dropout(0.2))

model.add(Dense( 512  , activation='sigmoid' ))

model.add(Dropout(0.2))

model.add(Dense( 256   , activation='sigmoid'))

model.add(Dropout(0.2))

model.add(Dense(128   , activation='sigmoid'))

model.add(Dropout(0.2))

model.add(Dense( 64   , activation='sigmoid'))

model.add(Dropout(0.2))

model.add(Dense( 5, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history = model.fit(x_train,y_train, epochs = 10, batch_size=32, validation_data = (x_test,y_test)  )
import matplotlib.pyplot as plt

h = history

plt.plot(h.history['accuracy'])

plt.plot(h.history['val_accuracy'])

plt.title('Model accuracy')

plt.show()



plt.plot(h.history['loss'])

plt.plot(h.history['val_loss'])

plt.title('Model Loss')

plt.show()
sample_data = "actor injured while shooting and the movie got cancelled"

processed_data = preprocess(sample_data)

x_sample = count_vectorizer.transform([processed_data])



# get results

model_result = list(model.predict(x_sample)[0])



print(model_result)



# get labels

max_ = max(model_result)

index = model_result.index(max_)

result = label_encoder.inverse_transform([index])



print("Model Result:",result)
sample_data = "Issues in china after the government tries to introduce a new law system "

processed_data = preprocess(sample_data)

x_sample = count_vectorizer.transform([processed_data])



# get results

model_result = list(model.predict(x_sample)[0])



print(model_result)



# get labels

max_ = max(model_result)

index = model_result.index(max_)

result = label_encoder.inverse_transform([index])



print("Model Result:",result)