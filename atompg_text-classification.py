# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import re

import spacy

from spacy.lang.en import English

from spacy.lang.en.stop_words import STOP_WORDS

import string

nlp = spacy.load('en_core_web_sm')

from sklearn.model_selection import train_test_split
df=pd.read_csv("../input/sentiment140/training.1600000.processed.noemoticon.csv",encoding="ISO-8859-1")

df_tweets=df.iloc[:,[0,5]]

df_tweets.columns=["label","tweet"]

df_small=df_tweets.iloc[789999:809999 ]



# df_tweets["label"]=df_tweets["label"].apply( lambda x:1 if x==4 else 0)

df_small["label"]=df_small["label"].apply( lambda x:1 if x==4 else 0)
df_tweets.sample(10)
# df_tweets["label"].value_counts()

df_small["label"].value_counts()
# df_tweets[df_tweets["label"]==0]

df_small[df_small["label"]==0]
# # Check for whitespace strings 

# blanks = []  # start with an empty list



# for i,lb,rv in df_tweets.itertuples():  

#     if type(rv)==str:            # avoid NaN values

#         if rv.isspace():         # test 'review' for whitespace

#             blanks.append(i)     # add matching index numbers to the list

        

# print(len(blanks))

# df_tweets.drop(blanks, inplace=True)

# Check for whitespace strings 

blanks = []  # start with an empty list



for i,lb,rv in df_small.itertuples():  

    if type(rv)==str:            # avoid NaN values

        if rv.isspace():         # test 'review' for whitespace

            blanks.append(i)     # add matching index numbers to the list

        

print(len(blanks))

df_small.drop(blanks, inplace=True)
def clean_text(text):

    text=text.lower()

    text = re.sub('@[A-Za-z0-9]+', '', text) #Removing @mentions

    text = re.sub('#', '', text) # Removing '#' hash tag

    text = re.sub('RT[\s]+', '', text) # Removing Retweets to avoid repitition

    text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink

    return text

# df_tweets["tweet"]=df_tweets["tweet"].apply(clean_text)

df_small["tweet"]=df_small["tweet"].apply(clean_text)
def clean(text):

    doc=nlp(text)

    lemmatized_tokens=[token.lemma_ for token in doc if token.is_stop==False and token.text not in string.punctuation and token.lemma_ !='-PRON-']

    cleaned_text= ' '.join(lemmatized_tokens)

    return cleaned_text
def stopwords(text):

    doc=nlp(text)

    stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',

             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',

             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',

             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 

             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',

             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',

             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',

             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',

             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',

             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',

             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',

             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 

             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',

             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',

             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",

             "youve", 'your', 'yours', 'yourself', 'yourselves']

    filtered=[]

    for token in doc:

        if token.text not in stopwordlist:

            filtered.append(token.text)

    return ' '.join(filtered)
df_small["tweet"]=df_small["tweet"].apply(clean)
X = df_small['tweet']

y = df_small['label']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics

from sklearn.svm import LinearSVC





text_clf = Pipeline([('tfidf', TfidfVectorizer()),

                     ('clf', LinearSVC()),

])



# Feed the training data through the pipeline

text_clf.fit(X_train, y_train)  

predictions_svc = text_clf.predict(X_test)



print(metrics.confusion_matrix(y_test,predictions_svc))

print("\n\n")

print(metrics.classification_report(y_test,predictions_svc))
from sklearn.naive_bayes import MultinomialNB

text_clf = Pipeline([('tfidf', TfidfVectorizer()),

                     ('clf', MultinomialNB()),

])

text_clf.fit(X_train, y_train) 

predictions_nb = text_clf.predict(X_test)

print("Confusion Matrix::")

print(metrics.confusion_matrix(y_test,predictions_nb))



print("\n\n")

print("Classificaton Report::")

print(metrics.classification_report(y_test,predictions_nb))