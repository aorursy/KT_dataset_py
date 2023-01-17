# References : 

# https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove/data#Data-Cleaning

# https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert/notebook#4.-Embeddings-and-Text-Cleaning

# https://www.kaggle.com/rerere/disaster-tweets-svm#TF-IDF
# Lib

# General

import os

import pandas as pd

import numpy as np

# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

# Text processing

import spacy

import string

import re

from nltk.stem.porter import PorterStemmer

# TF-IDF/SVM

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC

from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import GridSearchCV,train_test_split

# Handle warnings

import warnings

warnings.filterwarnings("ignore")

# import

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")
# NaN in df

f, axes = plt.subplots(1, 2)

plt.subplots_adjust(right = 1.8)

sns.barplot(['NaN','Samples'],[train.keyword.isna().sum(),train.keyword.count()],

            ax=axes[0]).set_title('keyword')

sns.barplot(['NaN','Samples'],[train.location.isna().sum(),train.location.count()],

            ax=axes[1]).set_title('location')
# Fill NaN with "else"

train.keyword.fillna("else", inplace = True) 

train.location.fillna("else", inplace = True) 

test.keyword.fillna("else", inplace = True) 



# Count unique words

kw = list(train.keyword.value_counts()).count(1)

loc = list(train.location.value_counts()).count(1)

sns.barplot(['keyword ({}/{})'.format(kw,train.keyword.count()),

             'location ({}/{})'.format(loc,train.location.count())]

            ,[kw,loc]).set_title('Unique values')
# Possible features

nlp = spacy.load('en')

def w_params(tar):

    str_len = tar.str.len()

    num_wor = tar.str.split().map(lambda x: len(x))

    uniq_wor = tar.apply(lambda x: len(set(str(x).split())))

    awl = tar.apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    stop_wor = tar.apply(lambda x: len([w for w in nlp(x) if w.is_stop == True]))

    pun = tar.apply(lambda x: len([p for p in str(x) if p in string.punctuation]))

    return [str_len,num_wor,uniq_wor,awl,stop_wor,pun]

train_text_features = w_params(train.text)

test_text_features = w_params(test.text)
# NLP

stm = PorterStemmer()

w_update = list(train.text)

test_update = list(test.text)

nlp = spacy.load('en')

def tp(tp_label):

    emoji = re.compile("["

                        u"\U0001F600-\U0001F64F"  # emoticons

                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                        u"\U0001F680-\U0001F6FF"  # transport & map symbols

                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                        u"\U00002702-\U000027B0"

                        u"\U000024C2-\U0001F251"

                        "]+", flags=re.UNICODE)

    url = re.compile(r'https?://\S+|www\.\S+')

    html = re.compile(r'<.*?>')

    nw_label = []

    for el in tp_label:

        doc = nlp(el)

        new_el = []

        for token in doc:

            if token.is_stop == False:

                rs = token.lemma_

                rs = stm.stem(rs)

                if rs not in string.punctuation and rs.isalpha():

                    rs = re.sub('[@#.;):):D\x89ûò\n\x89ÛÓåÊ\x89Û]', '', rs)

                    rs = emoji.sub(r'', rs)

                    rs = url.sub(r'', rs)

                    rs = html.sub(r'', rs)

                    new_el.append(rs) 

        nw_label.append(' '.join(word for word in new_el))

    return nw_label

w_update = tp(w_update)

test_update = tp(test_update)
# TF-IDF

tfidf = TfidfVectorizer(stop_words = 'english')

X_all = pd.concat([pd.Series(w_update), pd.Series(test_update)])

tfidf.fit(X_all)

X = tfidf.transform(w_update)

Xt = tfidf.transform(test_update)

# Parameters tuning

X_train, X_val, y_train, y_val = train_test_split(X, train.target, test_size=0.2, random_state=0)

parameters = { 

    'gamma': [0.01, 0.1, 0.3, 0.5, 0.7, 0.85, 1,'scale'], 

    'kernel': ['rbf', 'sigmoid'], 

    'C': [0.03, 0.1, 0.5, 1, 1.5, 2, 3.5, 5],

}

model_par = GridSearchCV(SVC(), parameters, cv=10, n_jobs=-1).fit(X_train, y_train)

model_par.cv_results_['params'][model_par.best_index_]
# SVM

bp = model_par.cv_results_['params'][model_par.best_index_]

model = SVC(C = bp['C'], gamma = bp['gamma'], kernel = bp['kernel'])



clf = CalibratedClassifierCV(model)

clf.fit(X, train.target)

mot = clf.predict_proba(Xt)

mot2 = clf.predict(Xt)
# Save to csv

output = pd.DataFrame({'Id': test.id,

                       'target': mot2})

output.to_csv('submission.csv', index=False)