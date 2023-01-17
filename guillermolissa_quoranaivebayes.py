%matplotlib inline 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import random

from nltk import word_tokenize, sent_tokenize

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
# Lematization
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
stoplist = stopwords.words("english")

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


train = train[train.qid.isin(['527aac2ce6f12f789fe5','7cd3188b81cdec72ad91'])!=True]

# Insincere question
qcorpus = [] # corpus

for text in train.question_text:
    tokens = []
    for token in word_tokenize(text):
        if token.isalpha() and token not in stoplist: # delete  stopwords and non characters
            token = wordnet_lemmatizer.lemmatize(token.lower()) # lematization token
            tokens.append(token)
    
    qcorpus.append(" ".join(tokens))
train['question_text'] = qcorpus

# Insincere question
qcorpus = [] # corpus

for text in test.question_text:
    tokens = []
    for token in word_tokenize(text):
        if token.isalpha() and token not in stoplist: # delete  stopwords and non characters
            token = wordnet_lemmatizer.lemmatize(token.lower()) # lematization token
            tokens.append(token)
    
    qcorpus.append(" ".join(tokens))
test['question_text'] = qcorpus
# incluyo bigramas (aclaracion:si aparece "best friend" tambien va a contar para "best" y para "friend")
count_vect = CountVectorizer(min_df=5,stop_words="english",ngram_range=(1,2))
X_train_ngrams = count_vect.fit_transform(train['question_text'] ) # cuenta frecuencia de tokens y define el diccionario
X_test_ngrams = count_vect.transform(test['question_text'] ) # cuenta frecuencia de tokens existentes en el diccionario
print("numero de features=",X_train_ngrams.shape[1])



clf = MultinomialNB(alpha=1)
clf.fit(X_train_ngrams, train.target)
test['prediction'] = clf.predict(X_test_ngrams)

submission = test[['qid','prediction']]
submission.to_csv('submission.csv',index=False,sep=",")