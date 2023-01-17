# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import nltk

import seaborn as sns

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import RFE

import re

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
train = pd.read_csv('/kaggle/input/tamil-nlp/tamil_news_train.csv')

test = pd.read_csv('/kaggle/input/tamil-nlp/tamil_news_test.csv')
train.head()

test.head()

train.shape

test.shape
train.isnull().any()

test.isnull().any()
train = train.drop_duplicates().reset_index(drop=True)

test = test.drop_duplicates().reset_index(drop=True)

train.shape

test.shape
stop_words = stopwords.words('english')



train.NewsInEnglish = train.NewsInEnglish.str.lower()

train.NewsInEnglish = train.NewsInEnglish.str.replace('[^\w\s]',' ')

train.NewsInEnglish = train.NewsInEnglish.str.replace('\d+', ' ')

train.NewsInEnglish = train.NewsInEnglish.str.replace(' txt', '')

train.NewsInEnglish = train.NewsInEnglish.apply(lambda x: [item for item in [x] if item not in stop_words])

count = 0

for i in range(len(train.NewsInEnglish)):

    train.NewsInEnglish[count] = train.NewsInEnglish[count][0].replace('  ', '')

#     train.NewsInEnglish[count] = train.NewsInEnglish[count][0].split(' ')

    count = count + 1

    

train.NewsInTamil = train.NewsInTamil.str.lower()

train.NewsInTamil = train.NewsInTamil.str.replace('[^\w\s]',' ')

train.NewsInTamil = train.NewsInTamil.str.replace('\d+', ' ')



test.NewsInTamil = test.NewsInTamil.str.lower()

test.NewsInTamil = test.NewsInTamil.str.replace('[^\w\s]',' ')

test.NewsInTamil = test.NewsInTamil.str.replace('\d+', ' ')



test.NewsInEnglish = test.NewsInEnglish.str.lower()

test.NewsInEnglish = test.NewsInEnglish.str.replace('[^\w\s]',' ')

test.NewsInEnglish = test.NewsInEnglish.str.replace('\d+', ' ')

test.NewsInEnglish = test.NewsInEnglish.str.replace(' txt', '')

test.NewsInEnglish = test.NewsInEnglish.apply(lambda x: [item for item in [x] if item not in stop_words])    

count = 0

for i in range(len(test.NewsInEnglish)):

    test.NewsInEnglish[count] = test.NewsInEnglish[count][0].replace('  ', '')

#     test.NewsInEnglish[count] = test.NewsInEnglish[count][0].split(' ')

    count = count + 1

    

train = train.append(test)

df = train

df.head()

df.shape
vect = TfidfVectorizer(ngram_range=(1,1),stop_words='english',min_df=100)

dtm = vect.fit_transform(df.NewsInEnglish)

dtm.shape
train_features, test_features, train_labels, test_labels = train_test_split(dtm, df['Category'], test_size=.20)
gnb = GaussianNB()

gnb.fit(train_features.toarray(), train_labels)

nb_pred_train = gnb.predict(train_features.toarray())

nb_pred_test = gnb.predict(test_features.toarray())
def multiclass_roc_auc_score(truth, pred, average="macro"):

    lb = LabelBinarizer()

    lb.fit(truth)



    truth = lb.transform(truth)

    pred = lb.transform(pred)



    return roc_auc_score(truth, pred, average=average)
print(classification_report(test_labels, nb_pred_test))

print('Naive Bayes baseline: ' + str(multiclass_roc_auc_score(train_labels, nb_pred_train)))

print('Naive Bayes: ' + str(multiclass_roc_auc_score(test_labels, nb_pred_test)))
vect = TfidfVectorizer(ngram_range=(1,1), min_df=100)

dtm = vect.fit_transform(df.NewsInTamil)

vect.get_feature_names()

dtm.shape
train_features, test_features, train_labels, test_labels = train_test_split(dtm, df['CategoryInTamil'], test_size=.20)
gnb = GaussianNB()

gnb.fit(train_features.toarray(), train_labels)

nb_pred_train = gnb.predict(train_features.toarray())

nb_pred_test = gnb.predict(test_features.toarray())
print(classification_report(test_labels, nb_pred_test))

print('Naive Bayes baseline: ' + str(multiclass_roc_auc_score(train_labels, nb_pred_train)))

print('Naive Bayes: ' + str(multiclass_roc_auc_score(test_labels, nb_pred_test)))