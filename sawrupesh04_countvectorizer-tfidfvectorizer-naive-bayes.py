# imports 

import numpy as np # linear algebra

import pandas as pd # data processing



# visualization

import matplotlib.pyplot as plt





from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS



# model selection

from sklearn.model_selection import train_test_split



# accuracy score

from sklearn.metrics import accuracy_score



# NLP

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



# Model

from sklearn.naive_bayes import MultinomialNB

# load the data

data = pd.read_csv('../input/nlp-starter-test/social_media_clean_text.csv')
# print head of the dataframe

data.head()
# print data info

data.info()
stopwords = set(STOPWORDS)
def wordcloud_plot(name_of_feature):

    plt.figure(figsize=(10, 10))

    wordcloud = WordCloud(

                              background_color='black',

                              stopwords=stopwords,

                              max_words=200,

                              max_font_size=40, 

                              random_state=42

                             ).generate(str(name_of_feature))

    fig = plt.figure(1)

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
relevant_text = data[data['choose_one']=='Relevant']['text']

irrelevant_text = data[data['choose_one']=='Not Relevant']['text']
wordcloud_plot(relevant_text)
wordcloud_plot(irrelevant_text)
# Sample dataset

X = data['text']

y = data['choose_one']
# Intialize CountVectorizer

cv = CountVectorizer()
# fit and transform CountVectorizer 

X1 = cv.fit_transform(X)
# Create train and test split with test size 33 %

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.33, random_state=53)
# Intiate Naive_bayes

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)
clf_score = accuracy_score(y_pred_clf, y_test)

print('Accuracy with CountVectorizer : ', clf_score*100)
# initiate TfidfVEctorizer

tv = TfidfVectorizer()
# fit and tranform the tfidfVectorizer

X2 = tv.fit_transform(X)
# create train and test split 

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.33, random_state=53)
# initialize Naive Bayes

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)
score = accuracy_score(y_pred_clf, y_test)

print('Accuracy with TfidfVectorizer : ', score*100)
text = data['text'].iloc[0]
# create Tfidf transform

temp = tv.transform([text])
clf.predict(temp)[0]