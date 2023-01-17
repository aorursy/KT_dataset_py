# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame, Series

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import  re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv").fillna('unknown')
test = pd.read_csv("../input/test.csv").fillna('unknown')
train.head()
train.info()
train['comment_text'][0]
train['comment_text'][3]
train['comment_text'][4]
#we could see that the length of the comments varies a lot
length = train.comment_text.str.len()
length.mean(), length.std(), length.max()
length.hist()
class_names = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
#comments without any label
train['none'] = 1-train[class_names].max(axis=1)
train.describe()
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
from scipy.sparse import hstack
from sklearn.pipeline import make_union
repl = {
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " frown ",
    ":(": " frown ",
    ":s": " frown ",
    ":-s": " frown ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
}
new_train_data = []
new_test_data = []

list_train = train['comment_text'].tolist()
list_test = test['comment_text'].tolist()

for i in list_train:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in repl.keys():
            j = repl[j]
        xx = xx + j + " "
    new_train_data.append(xx)

for i in list_test:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in repl.keys():
            j = repl[j]
        xx = xx + j + " "
    new_test_data.append(xx)

train["clean_comment_text"] = new_train_data
test["clean_comment_text"] = new_test_data
pattern = re.compile(r'[^a-zA-Z ?!]+')
train_text = train["clean_comment_text"].tolist()
test_text = test["clean_comment_text"].tolist()
for i,c in enumerate(train_text):
    train_text[i] = pattern.sub('',train_text[i].lower())
for i,c in enumerate(test_text):
    test_text[i] = pattern.sub('',test_text[i].lower())
train['comment_text'] = train_text
test["comment_text"] = test_text
del train_text, test_text
train.drop(['clean_comment_text'], inplace = True, axis = 1)
test.drop(['clean_comment_text'], inplace = True, axis = 1)

all_text = pd.concat([train['comment_text'],test['comment_text']])
word_vectorizer = TfidfVectorizer(ngram_range =(1,3),
                             tokenizer=tokenize,
                             min_df=3, max_df=0.9,
                             strip_accents='unicode',
                             stop_words = 'english',
                             analyzer = 'word',
                             use_idf=1,
                             smooth_idf=1,
                             sublinear_tf=1 )
char_vectorizer = TfidfVectorizer(ngram_range =(1,4),
                                 min_df=3, max_df=0.9,
                                 strip_accents='unicode',
                                 analyzer = 'char',
                                 stop_words = 'english',
                                 use_idf=1,
                                 smooth_idf=1,
                                 sublinear_tf=1,
                                 max_features=50000)
vectorizer = make_union(word_vectorizer, char_vectorizer)
vectorizer.fit(all_text)

train_matrix =vectorizer.transform(train['comment_text'])
test_matrix = vectorizer.transform(test['comment_text'])
train_matrix,test_matrix
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
val_score = []
def cross_validation(model,y_train):
    score = cross_val_score(model,train_matrix,y_train,scoring='accuracy',cv=5)
    val_score.append(score.mean())

class_names = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
model = MultinomialNB()
for clas in class_names:
    print(clas)
    cross_validation(model,train[clas])
NB_score = val_score
NB_score
val_score = []
LR_model = LogisticRegression(C=3, dual=True)
for clas in class_names:
    print(clas)
    cross_validation(LR_model,train[clas])
LR_score = val_score
LR_score
DF_score = pd.DataFrame(index=class_names)
DF_score['NB'] = NB_score
DF_score['LR'] = LR_score
DF_score
'''predictions = pd.DataFrame()
model = MultinomialNB()
for clas in class_names:
    model.fit(train_matrix,train[clas])
    prediction[clas] = model.predict(test_matrix)'''

def pr(y_i, y):
    p = train_matrix[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)
def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=3, dual=True)
    x_nb = train_matrix.multiply(r)
    return m.fit(x_nb, y), r
model = LogisticRegression(C=3,dual = True)
NBLR_score=[]
for clas in class_names:
    print(clas)
    y = train[clas].values
    r = np.log(pr(1,y) / pr(0,y))
    x_nb = train_matrix.multiply(r)
    score = cross_val_score(model,x_nb,y,scoring='accuracy',cv=5)
    NBLR_score.append(score.mean())
DF_score = pd.DataFrame(index=class_names)
DF_score['NB'] = NB_score
DF_score['LR'] = LR_score
DF_score['NBLR'] = NBLR_score
DF_score
preds = np.zeros((len(test), len(class_names)))

for i, j in enumerate(class_names):
    print('fit', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_matrix.multiply(r))[:,1]
submid = pd.DataFrame({'id': test["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = class_names)], axis=1)
submission.to_csv('submission_1.csv', index=False)
