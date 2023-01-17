# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from wordcloud import WordCloud
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import re,gc
from string import punctuation
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import SnowballStemmer,WordNetLemmatizer
stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()
import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
pd.set_option('max_colwidth',400)

train = pd.read_csv('../input/train.tsv', sep="\t")
test = pd.read_csv('../input/test.tsv', sep="\t")
sub = pd.read_csv('../input/sampleSubmission.csv', sep=",")
train.head(10)
print('Average count of phrases per sentence in train is {0:.0f}.'.format(train.groupby('SentenceId')['Phrase'].count().mean()))
print('Average count of phrases per sentence in test is {0:.0f}.'.format(test.groupby('SentenceId')['Phrase'].count().mean()))
print('Number of phrases in train: {}. Number of sentences in train: {}.'.format(train.shape[0], len(train.SentenceId.unique())))
print('Number of phrases in test: {}. Number of sentences in test: {}.'.format(test.shape[0], len(test.SentenceId.unique())))
print('Average word length of phrases in train is {0:.0f}.'.format(np.mean(train['Phrase'].apply(lambda x: len(x.split())))))
print('Average word length of phrases in test is {0:.0f}.'.format(np.mean(test['Phrase'].apply(lambda x: len(x.split())))))
text = ' '.join(train.loc[train.Sentiment == 4, 'Phrase'].values)
text_trigrams = [i for i in ngrams(text.split(), 3)]
text = ' '.join(train.loc[train.Sentiment == 4, 'Phrase'].values)
text = [i for i in text.split() if i not in stopwords.words('english')]
text_trigrams = [i for i in ngrams(text, 3)]
Counter(text_trigrams).most_common(30)
def clean_review(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z]',' ',review)
        #review=[stemmer.stem(w) for w in word_tokenize(str(review).lower())]
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus
train['csen']=clean_review(train.Phrase.values)
test['csen']=clean_review(test.Phrase.values)
y = train['Sentiment']
xtrain, xvalid, ytrain, yvalid = train_test_split(train.csen.values, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)
vectorizer = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
full_text = list(train['csen'].values)
vectorizer.fit(full_text)
xtrain_tfv =  vectorizer.transform(xtrain)
xvalid_tfv = vectorizer.transform(xvalid)
xtest_tfv = vectorizer.transform(test['csen'].values)
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))
lr = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))

xgb = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()
clf = MultinomialNB()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))
mnb = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()
clf = AdaBoostClassifier()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))
adboost = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()
clf = KNeighborsClassifier()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))
knc = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()
clf = LinearSVC()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))
lsvc = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()
clf = GradientBoostingClassifier()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))
gbc = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()
clf = ExtraTreesClassifier()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))
etc = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()
clf = DecisionTreeClassifier()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))
dtc = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()
sub['Sentiment'] = pd.DataFrame(lr)
sub.to_csv('lr.csv',index=False)
sub['Sentiment'] = pd.DataFrame(xgb)
sub.to_csv('xgb.csv',index=False)
sub['Sentiment'] = pd.DataFrame(mnb)
sub.to_csv('mnb.csv',index=False)
sub['Sentiment'] = pd.DataFrame(lsvc)
sub.to_csv('lsvc.csv',index=False)
sub['Sentiment'] = pd.DataFrame(etc)
sub.to_csv('etc.csv',index=False)
sub['Sentiment'] = pd.DataFrame(knc)
sub.to_csv('knc.csv',index=False)
sub['Sentiment'] = pd.DataFrame(dtc)
sub.to_csv('dtc.csv',index=False)
df = pd.DataFrame(lr,columns=['lr'])
df['xgb'] = xgb
df['mnb'] = mnb
df['lsvc'] = lsvc
df['etc'] = etc
df['knc'] = knc
df['dtc'] = dtc
df.head(15)
sub['Sentiment'] = df.mode(axis=1)
sub['Sentiment'] = sub.Sentiment.astype(int)
sub.to_csv('submission.csv',index=False)