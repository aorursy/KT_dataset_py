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
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')
sample.columns
import nltk
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from sklearn.naive_bayes import MultinomialNB 
import matplotlib.pyplot as plt
import string
#FROM https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc

## Number of unique words in the text ##
df_train["num_words"] = df_train["question_text"].apply(lambda x: len(str(x).split()))
df_test["num_words"] = df_test["question_text"].apply(lambda x: len(str(x).split()))
                                                                    
## Number of characters in the text ##
df_train["num_chars"] = df_train["question_text"].apply(lambda x: len(str(x)))
df_test["num_chars"] = df_test["question_text"].apply(lambda x: len(str(x)))
                                                                    
## Number of stopwords in the text ##
df_train["num_stopwords"] = df_train["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
df_test["num_stopwords"] = df_test["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
                                                                    
## Number of punctuations in the text ##
df_train["num_punctuations"] =df_train['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
df_test["num_punctuations"] =df_test['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
                  
## Number of title case words in the text ##
df_train["num_words_title"] = df_train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
df_test["num_words_title"] = df_test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
          
## Average length of the words in the text ##
df_train["mean_word_len"] = df_train["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
df_test["mean_word_len"] = df_test["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))         
# countVector
from sklearn.feature_extraction.text import CountVectorizer
import scipy as sp
count_vectorizer = CountVectorizer(decode_error='ignore')
Y = df_train['target'].values

X = sp.sparse.hstack((count_vectorizer.fit_transform(df_train.question_text),df_train[['num_words', 'num_chars', 'num_stopwords', 'num_punctuations', 'num_words_title', 'mean_word_len']].values),format='csr')
X_columns=count_vectorizer.get_feature_names()+df_train[['num_words', 'num_chars', 'num_stopwords', 'num_punctuations', 'num_words_title', 'mean_word_len']].columns.tolist()
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print('Classification rate for NB: ', model.score(Xtrain, Ytrain))
print('Classification rate for NB: ', model.score(Xtest, Ytest))
from sklearn.metrics import confusion_matrix
test_prediction = model.predict(Xtest)
print(confusion_matrix(Ytest, test_prediction))
# To understand overall, how well the model predicts. 
from sklearn.metrics import classification_report
print(classification_report(Ytest, test_prediction))
# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

# predict probabilities
probs = model.predict_proba(Xtest)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(Ytest, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(Ytest, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
pyplot.show()
# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
# precision-recall curve and f1
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from matplotlib import pyplot

probs = model.predict_proba(Xtest)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model.predict(Xtest)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(Ytest, probs)
# calculate F1 score
f1 = f1_score(Ytest, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(Ytest, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the roc curve for the model
pyplot.plot(recall, precision, marker='.')
# show the plot
pyplot.show()
final = sp.sparse.hstack((count_vectorizer.transform(df_test.question_text),df_test[['num_words', 'num_chars', 'num_stopwords', 'num_punctuations', 'num_words_title', 'mean_word_len']].values),format='csr')
df_test['prediction'] = model.predict(final)
print(df_test.columns)
df_test = df_test[['qid', 'prediction']]
df_test.to_csv('submission.csv', index=False)