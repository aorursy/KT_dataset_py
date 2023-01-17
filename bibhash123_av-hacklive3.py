import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/hacklive-3-guided-hackathon-nlp/Train.csv')

test = pd.read_csv('/kaggle/input/hacklive-3-guided-hackathon-nlp/Test.csv')

tags = pd.read_csv('/kaggle/input/hacklive-3-guided-hackathon-nlp/Tags.csv')

sample_submission = pd.read_csv('/kaggle/input/hacklive-3-guided-hackathon-nlp/SampleSubmission.csv')
train.head(3)
TARGET_COLS = list(tags["Tags"])

TOPIC_COLS = train.iloc[:,2:6].columns
import re

import nltk 

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

from functools import lru_cache
stemmer = SnowballStemmer("english")
nltk.download('stopwords')
#load processed data

train = pd.read_csv('../input/my-data-hacklive/train.csv')

test = pd.read_csv('../input/my-data-hacklive/test.csv')
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multiclass import OneVsRestClassifier

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split

from scipy.sparse import csr_matrix

from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression

import random

random.seed(500)                            #setting a random seed
# train data is split into train and validation sets

X_train,X_val,Y_train,Y_val = train_test_split(train.iloc[:,1:6],train.loc[:,TARGET_COLS],test_size=0.2,random_state = 101)
# Utility function to obtain the best thresholds for predicted probability

def get_best_thresholds(true, preds):

    thresholds = [i/100 for i in range(100)]

    best_thresholds = []

    for idx in range(25):

        f1_scores = [f1_score(true[:, idx], (preds[:, idx] > thresh) * 1) for thresh in thresholds]

        best_thresh = thresholds[np.argmax(f1_scores)]

        best_thresholds.append(best_thresh)

    return best_thresholds
tfidf = TfidfVectorizer(max_features=10000)

tfidf.fit(list(train["ABSTRACT"])+list(test["ABSTRACT"]))

train_data = tfidf.transform(X_train["ABSTRACT"])

val_data = tfidf.transform(X_val["ABSTRACT"])

test_data = tfidf.transform(test["ABSTRACT"])



train_data = csr_matrix(np.concatenate((train_data.toarray(),X_train.loc[:,TOPIC_COLS]),axis=1).astype('float32'))

val_data = csr_matrix(np.concatenate((val_data.toarray(),X_val.loc[:,TOPIC_COLS]),axis=1).astype('float32'))

test_data = csr_matrix(np.concatenate((test_data.toarray(),test.loc[:,TOPIC_COLS]),axis=1).astype('float32'))
tfidf_bigram = TfidfVectorizer(max_features = 10000,ngram_range=(2,2))

tfidf_bigram.fit(list(train["ABSTRACT"])+list(test["ABSTRACT"]))

train_data_bigram = tfidf_bigram.transform(X_train["ABSTRACT"])

val_data_bigram = tfidf_bigram.transform(X_val["ABSTRACT"])

test_data_bigram = tfidf_bigram.transform(test["ABSTRACT"])



train_data_bigram = csr_matrix(np.concatenate((train_data_bigram.toarray(),X_train.loc[:,TOPIC_COLS]),axis=1).astype('float32'))

val_data_bigram = csr_matrix(np.concatenate((val_data_bigram.toarray(),X_val.loc[:,TOPIC_COLS]),axis=1).astype('float32'))

test_data_bigram = csr_matrix(np.concatenate((test_data_bigram.toarray(),test.loc[:,TOPIC_COLS]),axis=1).astype('float32'))
del train,X_train,X_val,test
clf_lgb = OneVsRestClassifier(LGBMClassifier(

                    n_estimators=100,

                    objective='binary',

                    verbose=1,

                    max_depth=18 ,

                    subsample=0.5, 

                    colsample_bytree=0.8,

                    reg_alpha=0,

                    reg_lambda=1

                    ))

                         

clf_lgb.fit(train_data,Y_train)
val_preds = clf_lgb.predict_proba(val_data)

val_preds_lgb = np.copy(val_preds)                        # copy for use in ensembling

best_thresholds = get_best_thresholds(Y_val.values,val_preds)



for i, thresh in enumerate(best_thresholds):

    val_preds[:, i] = (val_preds[:, i] > thresh) * 1



f1_score(Y_val.values,val_preds,average='micro')
test_preds = clf_lgb.predict_proba(test_data)

lgb_test_preds = np.copy(test_preds)                    # copy for use in ensembling



for i, thresh in enumerate(best_thresholds):

    test_preds[:, i] = (test_preds[:, i] > thresh) * 1

sub = pd.DataFrame(test_preds,columns=list(TARGET_COLS)).astype(int)

sample_submission.iloc[:,1:] = sub.values



sample_submission.to_csv('./submission.csv',index=False)

from IPython.display import FileLinks

FileLinks('./')
clf_lgb_bigram = OneVsRestClassifier(LGBMClassifier(

                    n_estimators=100,

                    objective='binary',

                    verbose=1,

                    max_depth=18 ,

                    subsample=0.5, 

                    colsample_bytree=0.8,

                    reg_alpha=0,

                    reg_lambda=1

                    ))

                         

clf_lgb_bigram.fit(train_data_bigram,Y_train)
val_preds = clf_lgb_bigram.predict_proba(val_data_bigram)

val_preds_lgb_bigram = np.copy(val_preds)                        # copy for use in ensembling

best_thresholds = get_best_thresholds(Y_val.values,val_preds)



for i, thresh in enumerate(best_thresholds):

    val_preds[:, i] = (val_preds[:, i] > thresh) * 1



f1_score(Y_val.values,val_preds,average='micro')
test_preds = clf_lgb_bigram.predict_proba(test_data_bigram)

lgb_test_preds_bigram = np.copy(test_preds)                    # copy for use in ensembling



for i, thresh in enumerate(best_thresholds):

    test_preds[:, i] = (test_preds[:, i] > thresh) * 1

sub = pd.DataFrame(test_preds,columns=list(TARGET_COLS)).astype(int)

sample_submission.iloc[:,1:] = sub.values



sample_submission.to_csv('./submission.csv',index=False)

from IPython.display import FileLinks

FileLinks('./')
clf_logreg = OneVsRestClassifier(LogisticRegression(C=5,n_jobs=-1))

clf_logreg.fit(train_data,Y_train)



val_preds = clf_logreg.predict_proba(val_data)

val_preds_logreg = np.copy(val_preds)



best_thresholds = get_best_thresholds(Y_val.values,val_preds)

for i, thresh in enumerate(best_thresholds):

    val_preds[:, i] = (val_preds[:, i] > thresh) * 1



f1_score(Y_val.values,val_preds,average='micro')
test_preds = clf_logreg.predict_proba(test_data)

logreg_test_preds =np.copy(test_preds)



for i, thresh in enumerate(best_thresholds):

    test_preds[:, i] = (test_preds[:, i] > thresh) * 1

sub = pd.DataFrame(test_preds,columns=list(TARGET_COLS)).astype(int)

sample_submission.iloc[:,1:] = sub.values



sample_submission.to_csv('./submission.csv',index=False)

from IPython.display import FileLinks

FileLinks('./')
clf_logreg_bigram = OneVsRestClassifier(LogisticRegression(C=5,n_jobs=-1))

clf_logreg_bigram.fit(train_data_bigram,Y_train)



val_preds = clf_logreg_bigram.predict_proba(val_data_bigram)

val_preds_logreg_bigram = np.copy(val_preds)



best_thresholds = get_best_thresholds(Y_val.values,val_preds)

for i, thresh in enumerate(best_thresholds):

    val_preds[:, i] = (val_preds[:, i] > thresh) * 1



f1_score(Y_val.values,val_preds,average='micro')
test_preds = clf_logreg_bigram.predict_proba(test_data_bigram)

logreg_test_preds_bigram =np.copy(test_preds)



for i, thresh in enumerate(best_thresholds):

    test_preds[:, i] = (test_preds[:, i] > thresh) * 1

sub = pd.DataFrame(test_preds,columns=list(TARGET_COLS)).astype(int)

sample_submission.iloc[:,1:] = sub.values



sample_submission.to_csv('./submission.csv',index=False)

from IPython.display import FileLinks

FileLinks('./')
val_preds = (val_preds_lgb+val_preds_lgb_bigram+val_preds_logreg+val_preds_logreg_bigram)/4

best_thresholds = get_best_thresholds(Y_val.values,val_preds)

for i, thresh in enumerate(best_thresholds):

    val_preds[:, i] = (val_preds[:, i] > thresh) * 1



f1_score(Y_val.values,val_preds,average='micro')
test_preds = (lgb_test_preds+lgb_test_preds_bigram+logreg_test_preds+logreg_test_preds_bigram)/4

for i, thresh in enumerate(best_thresholds):

    test_preds[:, i] = (test_preds[:, i] > thresh) * 1

sub = pd.DataFrame(test_preds,columns=list(TARGET_COLS)).astype(int)

sample_submission.iloc[:,1:] = sub.values



sample_submission.to_csv('./submission.csv',index=False)

from IPython.display import FileLinks

FileLinks('./')