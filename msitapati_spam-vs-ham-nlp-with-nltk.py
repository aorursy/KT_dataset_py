# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read Dataset



fullCorpus = pd.read_csv('../input/SMSSpamCollection', sep='\t', header=None)

fullCorpus.columns = ['label', 'body_text']



fullCorpus.head()
# Dataset shape (rows, columns)

len(fullCorpus), len(fullCorpus.columns)
# number of "spam" items vs "ham" items

len(fullCorpus[fullCorpus['label']=='spam']), len(fullCorpus[fullCorpus['label']=='ham'])
# check for null values

fullCorpus['label'].isnull().sum(), fullCorpus['body_text'].isnull().sum()
# Read in the raw text

rawData = open('../input/SMSSpamCollection').read()



# Print the raw data

rawData[0:500]
parsedData = rawData.replace('\t', '\n').split('\n')

parsedData[0:6]
labelList = parsedData[0::2]

textList = parsedData[1::2]
labelList[0:5]
textList[0:5]
print(labelList[0:5])

print(textList[0:5])
print(len(labelList))

print(len(textList))
print(labelList[-5:])

print(textList[0:5])
# correct version for reading file with correct data shape

fullCorpus = pd.DataFrame({

    'label': labelList[:-1],

    'body_list': textList

})



fullCorpus.head()
pd.set_option('display.max_colwidth', 100)



data = pd.read_csv('../input/SMSSpamCollection', sep='\t', header=None)

data.columns = ['label', 'body_text']



data.head()
import string



def remove_punct(text):

    text_nopunct = "".join([char for char in text if char not in string.punctuation])

    return text_nopunct
# make new "clean data" column

data['body_text_clean'] = data['body_text'].apply(lambda x: remove_punct(x))

data.head()
import re



def tokenize(text):

    tokens = re.split('\W+', text)

    return tokens
# make new "tokenized data" column

data['body_text_tokenized'] = data['body_text_clean'].apply(lambda x: tokenize(x.lower()))

data.head()
import nltk



stopword = nltk.corpus.stopwords.words('english')



def remove_stopwords(tokenized_list):

    text = [word for word in tokenized_list if word not in stopword]

    return text
# make new "nostop data" column

data['body_text_nostop'] = data['body_text_tokenized'].apply(lambda x: remove_stopwords(x))

data.head()
def wrangle_text(text):

    text = "".join([word for word in text if word not in string.punctuation])

    tokens = re.split('\W+', text)

    text = [word for word in tokens if word not in stopword]

    return text
ps = nltk.PorterStemmer()



def stemming(tokenized_text):

    text = [ps.stem(word) for word in tokenized_text]

    return text
# make new "stemmed words data" column

data['body_text_stemed'] = data['body_text_nostop'].apply(lambda x: stemming(x))

data.head()
wn = nltk.WordNetLemmatizer()



def lemmatizing(tokenized_text):

    text = [wn.lemmatize(word) for word in tokenized_text]

    return text
# make new "lemmatized words data" column

data['body_text_lemmatized'] = data['body_text_nostop'].apply(lambda x: lemmatizing(x))

data.head()
def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens = re.split('\W+', text)

    text = " ".join([ps.stem(word) for word in tokens if word not in stopword])

    return text



data['cleaned_text'] = data['body_text'].apply(lambda x: clean_text(x))

data.head()
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(analyzer=wrangle_text)

X_counts = count_vect.fit_transform(data['body_text'])

print(X_counts.shape)

print(count_vect.get_feature_names()[:20])
X_counts_df = pd.DataFrame(X_counts.toarray())

X_counts_df.head()
X_counts_df.columns = count_vect.get_feature_names()

X_counts_df.head()
from sklearn.feature_extraction.text import CountVectorizer



ngram_vect = CountVectorizer(ngram_range=(2,2))

X_counts = ngram_vect.fit_transform(data['cleaned_text'])

print(X_counts.shape)

print(ngram_vect.get_feature_names()[:20])
X_counts_df = pd.DataFrame(X_counts.toarray())

X_counts_df.columns = ngram_vect.get_feature_names()

X_counts_df.head()
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf_vect = TfidfVectorizer(analyzer=clean_text)

X_tfidf = tfidf_vect.fit_transform(data['body_text'])

print(X_tfidf.shape)

print(tfidf_vect.get_feature_names()[:20])
X_tfidf_df = pd.DataFrame(X_tfidf.toarray())

X_tfidf_df.columns = tfidf_vect.get_feature_names()

X_tfidf_df.head()
import string



def count_punct(text):

    count = sum([1 for char in text if char in string.punctuation])

    return round(count/(len(text) - text.count(" ")), 3)*100



data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))

data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))



data.head()
from sklearn.feature_extraction.text import TfidfVectorizer



# decision to use the TF-IDF vecorizer to account for token relevance

tfidf_vect = TfidfVectorizer(analyzer=clean_text)

X_tfidf = tfidf_vect.fit_transform(data['body_text'])



# new DF without labels

X_features = pd.concat([data['body_len'], 

                        data['punct%'], 

                        pd.DataFrame(X_tfidf.toarray())], 

                        axis=1)

X_features.head()
from sklearn.ensemble import RandomForestClassifier
print(dir(RandomForestClassifier))

print(RandomForestClassifier())
from sklearn.model_selection import KFold, cross_val_score
# only set n_jobs paramenter to -1 to enable building the individual decision trees in parallel.

rf = RandomForestClassifier(n_jobs=-1)

# Kfold, only hyperparameter is how many splits: how many folds in our cross-validation

k_fold = KFold(n_splits=5)



# show cross-validation scores of 5 parallel jobs

cross_val_score(rf, X_features, data['label'], cv=k_fold, scoring='accuracy', n_jobs=-1)
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.model_selection import train_test_split
# setting and performing train/test split

X_train, X_test, y_train, y_test = train_test_split(X_features, data['label'], test_size=0.2)
# select settings for calling the random forest classifier

rf = RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1)

# fit the model

rf_model = rf.fit(X_train, y_train)
# random forest allows for feature importances to be explored

sorted(zip(rf_model.feature_importances_, X_train.columns), reverse=True)[0:10]
# y-pred is an array of predictions for each of the elements in the test set

y_pred = rf_model.predict(X_test)

# tell the model to score y-labels searching for spam probability

precision, recall, fscore, support = score(y_test,             # pass in the actual y labels

                                           y_pred,             # pass in the predictions

                                           pos_label='spam',   # positive label (searched for) is spam

                                           average='binary')   # scoring setting
print('Precision: {} / Recall: {} / Accuracy: {}'.format(round(precision, 3),

                                                        round(recall, 3),

                                                        round((y_pred==y_test).sum() / len(y_pred),3))) #sum True(y_pred==y_test) results and divide that by the total length of the test set
# (copy) setting and performing train/test split

X_train, X_test, y_train, y_test = train_test_split(X_features, data['label'], test_size=0.2)
def train_RF(n_est, depth):

    # instanciate random forest classifier

    rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, n_jobs=-1)

    # call rf and fit train the model

    rf_model = rf.fit(X_train, y_train)

    # call the model and predict on X_test

    y_pred = rf_model.predict(X_test)

    # call saved predictions y_pred and print scores

    precision, recall, fscore, support = score(y_test, y_pred, pos_label='spam', average='binary')

    print('Est: {} / Depth: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(

        n_est, depth, round(precision, 3), round(recall, 3),

        round((y_pred==y_test).sum() / len(y_pred), 3)))
# building the grid search functionality

for n_est in [10, 50, 100]:

    for depth in [10, 20, 30, None]:

        train_RF(n_est, depth)
from sklearn.model_selection import GridSearchCV
# call tf-idf and stored that as X_tfidf_feat. To test which of these vectorizing frameworks works better. 

tfidf_vect = TfidfVectorizer(analyzer=clean_text)

X_tfidf = tfidf_vect.fit_transform(data['body_text'])

X_tfidf_feat = pd.concat([data['body_len'], data['punct%'], pd.DataFrame(X_tfidf.toarray())], axis=1)



# estimator and parameter grid

rf = RandomForestClassifier()

param = {'n_estimators': [10, 150, 300],

        'max_depth': [30, 60, 90, None]}



# construct CV object

gs = GridSearchCV(rf, param, cv=5, n_jobs=-1)

# train stored grid search object

gs_fit = gs.fit(X_tfidf_feat, data['label'])

#  wrapping this gs_fit.cv_results_ in pd.DataFrame

pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]
# run X_count and stored that as X_count_feat. To test which of these vectorizing frameworks works better. 

count_vect = CountVectorizer(analyzer=clean_text)

X_count = count_vect.fit_transform(data['body_text'])

X_count_feat = pd.concat([data['body_len'], data['punct%'], pd.DataFrame(X_count.toarray())], axis=1)



# estimator and parameter grid

rf = RandomForestClassifier()

param = {'n_estimators': [10, 150, 300],

        'max_depth': [30, 60, 90, None]}



# construct CV object

gs = GridSearchCV(rf, param, cv=5, n_jobs=-1)

# train stored grid search object

gs_fit = gs.fit(X_count_feat, data['label'])

#  wrapping this gs_fit.cv_results_ in pd.DataFrame

pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]
from sklearn.ensemble import GradientBoostingClassifier
print(dir(GradientBoostingClassifier))

print(GradientBoostingClassifier())
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_features, data['label'], test_size=0.2)
def train_GB(est, max_depth, lr):

    gb = GradientBoostingClassifier(n_estimators=est, max_depth=max_depth, learning_rate=lr)

    gb_model = gb.fit(X_train, y_train)

    y_pred = gb_model.predict(X_test)

    precision, recall, fscore, train_support = score(y_test, y_pred, pos_label='spam', average='binary')

    print('Est: {} / Depth: {} / LR: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(

        est, max_depth, lr, round(precision, 3), round(recall, 3), 

        round((y_pred==y_test).sum()/len(y_pred), 3)))
for n_est in [50, 100, 150]:

    for max_depth in [3, 7, 11, 15]:

        for lr in [0.01, 0.1, 1]:

            train_GB(n_est, max_depth, lr)
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
# TF-IDF

tfidf_vect = TfidfVectorizer(analyzer=clean_text)

X_tfidf = tfidf_vect.fit_transform(data['body_text'])

X_tfidf_feat = pd.concat([data['body_len'], data['punct%'], pd.DataFrame(X_tfidf.toarray())], axis=1)



gb = GradientBoostingClassifier()

param = {

    'n_estimators': [100, 150], 

    'max_depth': [7, 11, 15],

    'learning_rate': [0.1]

}



clf = GridSearchCV(gb, param, cv=5, n_jobs=-1)

cv_fit = clf.fit(X_tfidf_feat, data['label'])

pd.DataFrame(cv_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]
# CountVectorizer

count_vect = CountVectorizer(analyzer=clean_text)

X_count = count_vect.fit_transform(data['body_text'])

X_count_feat = pd.concat([data['body_len'], data['punct%'], pd.DataFrame(X_count.toarray())], axis=1)



gb = GradientBoostingClassifier()

param = {

    'n_estimators': [50, 100, 150], 

    'max_depth': [7, 11, 15],

    'learning_rate': [0.1]

}



clf = GridSearchCV(gb, param, cv=5, n_jobs=-1)

cv_fit = clf.fit(X_count_feat, data['label'])

pd.DataFrame(cv_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(data[['body_text', 'body_len', 'punct%']], data['label'], test_size=0.2)
# vectorize text

tfidf_vect = TfidfVectorizer(analyzer=clean_text)

tfidf_vect_fit = tfidf_vect.fit(X_train['body_text'])



tfidf_train = tfidf_vect_fit.transform(X_train['body_text'])

tfidf_test = tfidf_vect_fit.transform(X_test['body_text'])



X_train_vect = pd.concat([X_train[['body_len', 'punct%']].reset_index(drop=True), 

           pd.DataFrame(tfidf_train.toarray())], axis=1)

X_test_vect = pd.concat([X_test[['body_len', 'punct%']].reset_index(drop=True), 

           pd.DataFrame(tfidf_test.toarray())], axis=1)



X_train_vect.head()
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import precision_recall_fscore_support as score

import time
rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)



start = time.time()

rf_model = rf.fit(X_train_vect, y_train)

end = time.time()

fit_time = (end - start)



start = time.time()

y_pred = rf_model.predict(X_test_vect)

end = time.time()

pred_time = (end - start)



precision, recall, fscore, train_support = score(y_test, y_pred, pos_label='spam', average='binary')

print('Fit time: {} / Predict time: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(

    round(fit_time, 3), round(pred_time, 3), round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))
gb = GradientBoostingClassifier(n_estimators=150, max_depth=11)



start = time.time()

gb_model = gb.fit(X_train_vect, y_train)

end = time.time()

fit_time = (end - start)



start = time.time()

y_pred = gb_model.predict(X_test_vect)

end = time.time()

pred_time = (end - start)



precision, recall, fscore, train_support = score(y_test, y_pred, pos_label='spam', average='binary')

print('Fit time: {} / Predict time: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(

    round(fit_time, 3), round(pred_time, 3), round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))