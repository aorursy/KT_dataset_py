# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import nltk
import re
import string

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from nltk.tokenize import TweetTokenizer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# **** FUNCTIONS **************************************
# =====================================================    
def avg_word(sentence):
  words = sentence.split()
  if len(words) > 0:
      return (sum(len(word) for word in words)/len(words))
  else: 
      return 0    


# https://www.kaggle.com/saganandrij/xgbclassifier
def clean_text(text):
    #text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('[%s]' % re.escape(string.digits), '', text)
    text = re.sub('[%s]' % re.escape(' +'), ' ', text)
    text = text.replace(' n\'t', ' not')
    text = text.replace('it \'s', 'it is')
    text = text.replace('there \'s', 'there is')
    text = text.replace('he \'s', 'he is')
    text = text.replace('she \'s', 'she is')
    text = text.replace('what \'s', 'what is')
    text = text.replace('that \'s', 'that is')
    text = text.replace(' \'s', '')
    text = text.replace('\'s', '')
    text = text.replace('s \'', '')
    text = text.replace('-lrb-', '')
    text = text.replace('-LRB-', '')
    text = text.replace('-rrb-', '')
    text = text.replace('-RRB-', '')
    text = text.lower()
    text = text.strip()
    return text        



# #####################################################
# =====================================================    
# =====================================================

print("completed")
train = pd.read_csv('../input/train.tsv', sep="\t")
test = pd.read_csv('../input/test.tsv', sep="\t")
sub = pd.read_csv('../input/sampleSubmission.csv', sep=",")
sub.head()
sub.shape

#test['Sentiment']=-999
test.head()
test.shape
print("completed")
train.head()
train['Phrase_clean_text'] = train['Phrase'].apply(lambda x: clean_text(x))
test['Phrase_clean_text'] = test['Phrase'].apply(lambda x: clean_text(x))
print("completed")
train.head()
train_text = train['Phrase_clean_text'] # output Series
test_text = test['Phrase_clean_text'] # output Series
all_text = pd.concat([train_text, test_text]) # concat to 1 series with 222352 obs.

# le=LabelEncoder()
# y=le.fit_transform(train.Sentiment.values)
y = train["Sentiment"]

# dividing X, y into train and test data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
# X_train_vec, X_test_vec, y_train_vec, y_test_vec = train_test_split(train_vec_features, y, train_size=0.75, random_state = 42)
train_x, valid_x, train_y, valid_y = train_test_split(train['Phrase_clean_text'], y, train_size=0.75, random_state = 42)
print("completed")
# https://www.kaggle.com/tunguz/lr-with-words-and-char-n-grams
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode', # Remove accents during the preprocessing step. ‘ascii’ is a fast method that only works on characters that have an direct ASCII mapping. ‘unicode’ is a slightly slower method that works on any characters.
    analyzer='word', # Whether the feature should be made of word or character n-grams.
    token_pattern=r'\w{1,}', # Regular expression denoting what constitutes a “token”, only used if analyzer == 'word'. 
    stop_words='english', # If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens. Only applies if analyzer == 'word'.
    ngram_range=(1, 3), # The lower and upper boundary of the range of n-values for different n-grams to be extracted.
    max_features=300000)

vectorizer.fit(all_text)
xtrain_v =  vectorizer.transform(train_x) # transform the training and validation data using count vectorizer object
xvalid_v =  vectorizer.transform(valid_x)
x_unseen_v = vectorizer.transform(test_text)

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=20000)
count_vect.fit(all_text)
xtrain_count =  count_vect.transform(train_x) # transform the training and validation data using count vectorizer object
xvalid_count =  count_vect.transform(valid_x)
x_unseen_count = count_vect.transform(test_text)
print(xtrain_count.shape)
print(xvalid_count.shape)
print("x_unseen_count: ", x_unseen_count.shape)
#xtrain_count_df = pd.DataFrame(xtrain_count.toarray())
#xvalid_count_df = pd.DataFrame(xvalid_count.toarray())

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=15000)
tfidf_vect.fit(all_text)
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)
x_unseen_tfidf = tfidf_vect.transform(test_text)
print(xtrain_tfidf.shape)
print(xvalid_tfidf.shape)
#xtrain_tfidf_df = pd.DataFrame(xtrain_tfidf.toarray())
#xvalid_tfidf_df = pd.DataFrame(xvalid_tfidf.toarray())

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3), max_features=100000)
tfidf_vect_ngram.fit(all_text)
#xtrain_tfidf_ngram
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
x_unseen_tfidf_ngram = tfidf_vect_ngram.transform(test_text)
print(xtrain_tfidf_ngram.shape)
print(xvalid_tfidf_ngram.shape)



print("completed")
lsv = LinearSVC()
nb = MultinomialNB()
lr = LogisticRegression()
#lr = LogisticRegression(random_state=0)

lsv_model = lsv.fit(xtrain_v, train_y)
predictionslsv = lsv_model.predict(xvalid_v)

lr_model = lr.fit(xtrain_v, train_y)
predictionslr = lr_model.predict(xvalid_v)

#lsv_model = lsv.fit(xtrain_tfidf, train_y)
#predictionslsv = lsv_model.predict(xvalid_tfidf)

#lr_model = lr.fit(xtrain_tfidf, train_y)
#predictionslr = lr_model.predict(xvalid_tfidf)

print("LSV: " , accuracy_score(valid_y, predictionslsv))
print("LR: " , accuracy_score(valid_y, predictionslr))

predictions_test = lsv_model.predict(x_unseen_v)
predictions_test
sub.head()
sub.Sentiment=predictions_test
#sub.head()
sub.to_csv('submission0826_02.csv',index=False)
print("completed 2")
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

log_clf = LogisticRegression(random_state=42)
nb_clf = MultinomialNB()
#svm_clf = SVC(random_state=42)
lsv_clf = LinearSVC(random_state=42)
print("completed")
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('nb', nb_clf), ('lsv', lsv_clf)],
    voting='hard')

print("completed")
for clf in (log_clf, nb_clf, lsv_clf, voting_clf):
    clf.fit(xtrain_count, train_y)
    y_pred = clf.predict(xvalid_count)
    print(clf.__class__.__name__, accuracy_score(valid_y, y_pred))

            
print("completed")            
voting_clf_model = voting_clf.fit(xtrain_tfidf, train_y)
print("completed")
predictions_test = voting_clf_model.predict(x_unseen_tfidf)
predictions_test
sub.head()
sub.Sentiment=predictions_test
#sub.head()
sub.to_csv('submission0825_02.csv',index=False)
print("completed 2")
lsv_clf_model = lsv_clf.fit(xtrain_tfidf_ngram, train_y)
print("completed")
predictions_test = lsv_clf_model.predict(x_unseen_tfidf_ngram)
predictions_test
sub.head()
sub.Sentiment=predictions_test
#sub.head()
sub.to_csv('submission0825_03.csv',index=False)
print("completed 3")
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 200).fit(xtrain_count, train_y) # .74
predictions = dtree_model.predict(xvalid_count)
print(accuracy_score(valid_y, predictions))
sub.head()
sub.count()
predictions
predictions.shape