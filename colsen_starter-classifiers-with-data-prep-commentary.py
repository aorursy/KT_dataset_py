import numpy as np 
import pandas as pd 

from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem import SnowballStemmer,WordNetLemmatizer
stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()

from string import punctuation

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import FeatureUnion

from xgboost import XGBClassifier
train = pd.read_csv('../input/train.tsv', sep="\t")
test = pd.read_csv('../input/test.tsv', sep="\t")
sub = pd.read_csv('../input/sampleSubmission.csv', sep=",")
train.head()
def clean_text(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z]',' ',review)
        #review=[stemmer.stem(w) for w in word_tokenize(str(review).lower())]
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus

train['cleaned']=clean_text(train.Phrase.values)
train.head()

test['cleaned']=clean_text(test.Phrase.values)
tfhash = [("tfidf", TfidfVectorizer(stop_words='english')),
       ("hashing", HashingVectorizer (stop_words='english'))]
train_vectorized = FeatureUnion(tfhash).fit_transform(train.cleaned)
test_vectorized = FeatureUnion(tfhash).transform(test.cleaned)
# define outcome variable, y
y = train['Sentiment']

# train model
xgb = XGBClassifier()
xgb.fit(train_vectorized, y)
scores = cross_val_score(xgb, train_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)
print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))
logreg = LogisticRegression()
ovr = OneVsRestClassifier(logreg)
ovr.fit(train_vectorized, y)
scores = cross_val_score(ovr, train_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)
print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))
svm = LinearSVC()
svm.fit(train_vectorized, y)
scores = cross_val_score(svm, train_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)
print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))
#estimators = [ ('xgb',xgb) , ('ovr', ovr), ('svm',svm) ]
estimators = [ ('ovr', ovr), ('svm',svm) ]
clf = VotingClassifier(estimators , voting='soft')
clf.fit(train_vectorized,y)
#scores = cross_val_score(clf, train_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)
#print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))
#scores = clf.predict(train_vectorized)
#print(classification_report(scores, y))
#print(accuracy_score(scores, y))
#from sklearn.model_selection import train_test_split
#seed = 1234
#X_train, X_val, Y_train, Y_val = train_test_split(train_vectorized, y, test_size=0.25, random_state=seed)
sub['Sentiment'] = clf.predict(test_vectorized) 
sub.to_csv("clf.csv", index=False)