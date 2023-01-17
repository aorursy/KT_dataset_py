# Import the libraries

import pandas as pd

from sklearn.model_selection import train_test_split

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

import re

from nltk.tokenize import word_tokenize

from string import punctuation 

from nltk.corpus import stopwords 

# Downloads

nltk.download('stopwords')

nltk.download('punkt')
df = pd.read_csv('https://raw.githubusercontent.com/jonnybegreat/test-repo/master/twitter_train.csv')

test_df = pd.read_csv('https://raw.githubusercontent.com/jonnybegreat/test-repo/master/twitter_test.csv')
# Cleaning

'''

# Create a function to clean the tweets



def cleanTxt(text):

  

  

  text = re.sub('Ã¢â‚¬Â¦', '', text) #Removing @mentions

  #text = re.sub('#', '', text) # Removing '#' hash tag

  text = re.sub('RT[\s]+', '', text) # Removing RT

  text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink

  #text = re.sub(':', '', text) # Removing ':'

  #text = text.lower()

  #text = word_tokenize(text)

  

  return text





# Clean the tweets

df['message'] = df['message'].apply(cleanTxt)

test_df['message'] = test_df['message'].apply(cleanTxt)

# Show the cleaned tweets

df['message'][7]

'''
# Preparing

X = df['message'].astype(str)

y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=1)
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction import DictVectorizer
# Different Models

LogReg = LogisticRegression()

LinSVC = LinearSVC()

NB = MultinomialNB()

onevrest = OneVsRestClassifier(LinearSVC(),n_jobs=4)

vectorizer = TfidfVectorizer(

                             min_df=2, 

                             max_df=0.9,strip_accents='unicode',

                             analyzer='word',

                             ngram_range=(1, 2))
# Vecorize

tfidf = vectorizer.fit(X_train)

X_train = tfidf.transform(X_train)

X_test = tfidf.transform(X_test)
import pickle



pickle.dump(tfidf, open("tfidf.pkl", "wb"))
'''

# Create Pipeline

text_clf = Pipeline([('tfidf',TfidfVectorizer(

                             min_df=3, 

                             max_df=0.7,strip_accents='unicode',

                             analyzer='word',

                             ngram_range=(1, 3))),('clf',LinSVC)])

'''                             
# Fit Models

linsvcmodel = LinSVC.fit(X_train,y_train) #fits this pipeline using the training data

naivebayesmodel = NB.fit(X_train,y_train)

logisticregression = LogReg.fit(X_train,y_train)

# Pickling

pickle.dump(linsvcmodel, open("linsvcmodel.pkl", "wb"))

pickle.dump(naivebayesmodel, open("naivebayesmodel.pkl", "wb"))

pickle.dump(logisticregression, open("logisticregression.pkl", "wb"))
# Choose Model

text_clf = linsvcmodel
# Predict

predictions = text_clf.predict(X_test)
# Metrics

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score

from sklearn import metrics

print(classification_report(y_test,predictions))

print(f"Accuracy score : {accuracy_score(y_test,predictions)}")

print(f"f1 score : {f1_score(y_test,predictions,average='weighted')}")
# look at Confustion Matrix in more detail

import matplotlib.pyplot as plt

conf_mat = confusion_matrix(y_test, predictions)

fig, ax = plt.subplots(figsize=(4,4))

sns.heatmap(conf_mat, annot=True, fmt='d',

            xticklabels=['-1','0','1','2'], yticklabels=['-1','0','1','2'])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()
kaggle_predictions = text_clf.predict(vectorizer.transform(test_df["message"].astype(str)))
kaggle_df = pd.DataFrame(

    {'tweetid': test_df['tweetid'],

     'sentiment': kaggle_predictions

    })
kaggle_df.shape
kaggle_df.to_csv("sentiment_29.csv",index=False)