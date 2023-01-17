import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from wordcloud import WordCloud



# data from:

# https://www.kaggle.com/uciml/sms-spam-collection-dataset

# file contains some invalid chars

# depending on which version of pandas you have

# an error may be thrown

df = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', encoding= "ISO-8859-1")



# drop unnecessary columns

df = df.drop(["Unnamed: 2",

              "Unnamed: 3",

              "Unnamed: 4",], axis=1)

df.columns = ['labels', 'features']



# create binary labels

df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})

Y = df['b_labels'].to_numpy()



count_vectorizer = CountVectorizer(decode_error='ignore')

X1 = count_vectorizer.fit_transform(df['features'])

tfidf_vectorizer = TfidfVectorizer(decode_error='ignore')

X2 = tfidf_vectorizer.fit_transform(df['features'])



# split up the data

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X1, Y, test_size=0.33)

Xtrain, Xtest2, Ytrain, Ytest2 = train_test_split(X2, Y, test_size=0.33)





model= MultinomialNB()

model.fit(Xtrain, Ytrain)

print ("Test Score COUNT:", model.score(Xtest, Ytest))

print ("Test Score TFIDF:", model.score(Xtest2, Ytest2))



mess = ['you are a WINNER!! To claim 1000$ call 99999999']

output = model.predict(count_vectorizer.transform(mess))



for i ,m in enumerate(mess):

    print (m, ' == ', output[i])

model= LogisticRegression()

model.fit(Xtrain, Ytrain)

print ("Test Score COUNT:", model.score(Xtest, Ytest))

print ("Test Score TFIDF:", model.score(Xtest2, Ytest2))



mess = ['I shall be late to work today. I am sick.']

output = model.predict(count_vectorizer.transform(mess))



for i ,m in enumerate(mess):

    print (m, ' == ', output[i])

def visualize(label):

    words=''

    for msg in df[df['labels']== label]['features']:

                  msg=msg.lower()

                  words+=msg+" "

                  wordc=WordCloud(height=600, width=400).generate(words)

                  plt.imshow(wordc)

                  plt.axis('off')

                  plt.show()

                  

#visualize("ham")

#visualize("spam")