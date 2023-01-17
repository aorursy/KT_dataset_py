# Importing important libraries

import numpy as np

import pandas as pd



import re

from collections import defaultdict



from nltk.corpus import stopwords





# Printing files in input folder

import os

print(os.listdir("../input"))



from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import classification_report

from sklearn.utils import shuffle
# Loading data from CSV

data = pd.read_csv("../input/spam.csv", encoding = "latin-1")

data.head()
# Selecting and renaming first 2 columns

data = data[['v1','v2']]

data.columns = ['label','text']
# Visual data

data.head()
# Converting ham and spam to 0 and 1 respectively

data['label'] = data['label'].map({'ham':0,'spam':1})
# Printing number of ham and spam emails

data['label'].value_counts()
# Creating a new shuffled dataset with equal ham and spam emails

ham = data[data['label'] == 0]

spam = data[data['label'] == 1]

new_ham = ham.sample(len(spam), random_state = 5)

new_data = pd.concat([new_ham,spam],axis = 0)

data = shuffle(new_data, random_state = 5).reset_index(drop=True)
# Defining a text parsing function which will tokenize the text. It removes all punctuation, spaces, and stopwords

def textParser(text):

    tokens = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', "", text).split(" ")

    tokens = list(filter(lambda x: len(x) > 0 , map(str.lower,tokens)))

    tokens = list(filter(lambda x: x not in stopwords.words("english"),tokens))

    return tokens
# Converting each text into a vector format

bow_data = CountVectorizer(analyzer = textParser).fit_transform(data['text'])
# Normalizing the vectorized texts by text length

tfidf_data = TfidfTransformer().fit_transform(bow_data)
# Splitting the normalized, vectorized texts into train and test sets

X_train, X_test, Y_train, Y_test = train_test_split(tfidf_data,data[['label']], test_size=0.3, random_state = 5)
# Defining a Gaussian model

model = MultinomialNB()
# Fitting the model to the training data

fitted_model = model.fit(X_train.toarray(), np.array(Y_train).ravel())
# Predicting on the test data and printing the accuracy

pred = fitted_model.predict(X_test.toarray())

acc_MNB = accuracy_score(np.array(Y_test).ravel(), pred)

acc_MNB
# Printing the classification report

print(classification_report(np.array(Y_test).ravel(),pred))
# Creating the training pipeline

training_pipe = Pipeline(

    steps = [

        ('bow', CountVectorizer(analyzer = textParser)),

        ('tfdif', TfidfTransformer()),

        ('model',MultinomialNB())

    ]

)
# Creating training data from the unvectorized data

X_train, X_test, Y_train, Y_test = train_test_split(data['text'], data['label'], test_size = 0.3, random_state = 5)
# Fitting model and predicting on data

training_pipe.fit(X_train,Y_train)

pred_test_MNB = training_pipe.predict(X_test)

print("Accuracy (%):",training_pipe.score(X_test, Y_test)*100)