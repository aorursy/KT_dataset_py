# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Firstly we read our data

data = pd.read_csv('/kaggle/input/classify-text-by-buy-and-sell/buy_and_sell_text.csv',names=["Intent", "Text"])

out = pd.read_csv('/kaggle/input/globussoft-out-df/out.csv')
data.head()
#We must change category 1 or 0

data["Intent"] = [1 if each == "buy" else 0 for each in data["Intent"]]
data.head()
from sklearn.utils import shuffle

data = shuffle(data)
text = out["Message"]
#We choose 1 row.And we throw punctuation

import re

nlp_data = str(data.iloc[2,:])

nlp_out =str(out.iloc[2,:])

nlp_data = re.sub("[^a-zA-Z]"," ",nlp_data)

nlp_out = re.sub("[^a-zA-Z]"," ",nlp_out)
#After return lower case

nlp_data = nlp_data.lower()

nlp_out = nlp_out.lower()
#we have two choice we can use split methot or tokenize

import nltk as nlp

nlp_data = nlp.word_tokenize(nlp_data)



nlp_out = nlp.word_tokenize(nlp_out)

#nlp_data = nlp_data.split() or we can do so
#we have to find word root

lemma = nlp.WordNetLemmatizer()

nlp_data = [lemma.lemmatize(word) for word in nlp_data]

nlp_out = [lemma.lemmatize(word) for word in nlp_data]
#We join our data

nlp_data = " ".join(nlp_data)

nlp_out = " ".join(nlp_out)
import nltk as nlp

import re

description_list = []

for description in data["Text"]:

    description = re.sub("[^a-zA-Z]",' ',str(description))

    description = description.lower()   # buyuk harftan kucuk harfe cevirme

    description = nlp.word_tokenize(description)

    #description = [ word for word in description if not word in set(stopwords.words("english"))]

    lemma = nlp.WordNetLemmatizer()

    description = [ lemma.lemmatize(word) for word in description]

    description = " ".join(description)

    description_list.append(description) #we hide all word one section
description_list_out = []

for description in out["Message"]:

    description = re.sub("[^a-zA-Z]",' ',str(description))

    description = description.lower()   # buyuk harftan kucuk harfe cevirme

    description = nlp.word_tokenize(description)

    #description = [ word for word in description if not word in set(stopwords.words("english"))]

    lemma = nlp.WordNetLemmatizer()

    description = [ lemma.lemmatize(word) for word in description]

    description = " ".join(description)

    description_list_out.append(description) #we hide all word one section
#We make bag of word it is including number of all word's info

from sklearn.feature_extraction.text import CountVectorizer 

max_features = 3000 #We use the most common word

count_vectorizer = CountVectorizer(max_features = max_features, stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()

print("the most using {} words: {}".format(max_features,count_vectorizer.get_feature_names()))
from sklearn.feature_extraction.text import CountVectorizer 

max_features = 3000 #We use the most common word

count_vectorizer = CountVectorizer(max_features = max_features, stop_words = "english")

sparce_matrix_out = count_vectorizer.fit_transform(description_list_out).toarray()

print("the most using {} words: {}".format(max_features,count_vectorizer.get_feature_names()))
#We separate our data is train and test

y = data.iloc[:,0].values   # male or female classes

x = sparce_matrix

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.10, random_state = 42)
#We make model for predict

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("the accuracy of our model: {}".format(nb.score(x_test,y_test)))
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter = 200)

lr.fit(x_train,y_train)

predict_lr = lr.predict(x_test)

print(accuracy_score(predict_lr, y_test)*100)

print("our accuracy is: {}".format(lr.score(x_test,y_test)))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

predict_knn=knn.fit(x_train,y_train)

#print('Prediction: {}'.format(prediction))

print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test))
seg=lr.predict(sparce_matrix_out)
out['seg'] = seg
for i in out:

    if out[i].isnull().sum() >= 20000:

        out=out.drop([i], axis=1)
out.isnull().sum()
out['Message']
seg_out =out.groupby(['seg'])

#seg_out.sum().reset_index().to_csv('seg_out.csv')
seg_out