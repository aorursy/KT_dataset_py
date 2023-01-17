# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk 

from nltk import word_tokenize



from nltk.corpus import stopwords

import re

import string

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import normalize

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import SGDClassifier

from sklearn import metrics

from sklearn.metrics import f1_score,precision_score,recall_score

from sklearn import svm

from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS





from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/all.csv') #load dataset
df.head() #getting top 5 head
df.shape #getting shape
df.info() #info of dataset
df.isnull().sum() #checking again null vaules
df.groupby('type').count()
#looking in content

df['content']








stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='orange',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=15

                         ).generate(str(df[df['type']=='Mythology & Folklore']['content']))



fig = plt.figure(1,figsize=(12,18))

plt.imshow(wordcloud)

plt.title('Mythology & Folklore')

plt.axis('off')

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='orange',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=15

                         ).generate(str(df[df['type']=='Love']['content']))



fig = plt.figure(1,figsize=(12,18))

plt.title('Love')

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='orange',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=15

                         ).generate(str(df[df['type']=='Nature']['content']))



fig = plt.figure(1,figsize=(12,18))

plt.title('Nature')

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
labels = 'Love', 'Mythology & Folklore', 'Nature'

sizes = [326, 59, 188]





fig1, ax1 = plt.subplots(figsize=(5,5))

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
#function to remove ounctuation

def removePunctuation(x):

    x = x.lower()

    x = re.sub(r'[^\x00-\x7f]',r' ',x)

    x = x.replace('\r','')

    x = x.replace('\n','')

    x = x.replace('  ','')

    x = x.replace('\'','')

    return re.sub("["+string.punctuation+"]", " ", x)





#getting stop words

from nltk.corpus import stopwords



stops = set(stopwords.words("english")) 





#function to remove stopwords

def removeStopwords(x):

    filtered_words = [word for word in x.split() if word not in stops]

    return " ".join(filtered_words)





def processText(x):

    x= removePunctuation(x)

    x= removeStopwords(x)

    return x





from nltk.tokenize import sent_tokenize, word_tokenize

X= pd.Series([word_tokenize(processText(x)) for x in df['content']])

X.head()
#vectorizing X and y to process

vectorize=CountVectorizer(max_df=0.95, min_df=0.005)

X=vectorize.fit_transform(df['content'], df['author'])

vect = CountVectorizer(tokenizer = lambda x: x.split(), binary = 'true')

y = vect.fit_transform(df.type)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=42)





#sgd classifier



classifier = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.00001, penalty='l1'), n_jobs=-1)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)



print("accuracy :",classifier.score(X_train, y_train))

print("macro f1 score :",metrics.f1_score(y_test, predictions, average = 'macro'))

print("micro f1 scoore :",metrics.f1_score(y_test, predictions, average = 'micro'))

print("hamming loss :",metrics.hamming_loss(y_test,predictions))

print("Precision recall report :\n",metrics.classification_report(y_test, predictions))
#logistic regression



classifier = OneVsRestClassifier(LogisticRegression(penalty='l1'), n_jobs=-1)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)



print("accuracy :",classifier.score(X_train, y_train))

print("macro f1 score :",metrics.f1_score(y_test, predictions, average = 'macro'))

print("micro f1 scoore :",metrics.f1_score(y_test, predictions, average = 'micro'))

print("hamming loss :",metrics.hamming_loss(y_test,predictions))

print("Precision recall report :\n",metrics.classification_report(y_test, predictions))
#linear svc classifier

from sklearn.svm import LinearSVC



classifier = OneVsRestClassifier(LinearSVC(random_state=0, tol=1e-5), n_jobs=-1)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)



print("accuracy :",classifier.score(X_train, y_train))

print("macro f1 score :",metrics.f1_score(y_test, predictions, average = 'macro'))

print("micro f1 scoore :",metrics.f1_score(y_test, predictions, average = 'micro'))

print("hamming loss :",metrics.hamming_loss(y_test,predictions))

print("Precision recall report :\n",metrics.classification_report(y_test, predictions))