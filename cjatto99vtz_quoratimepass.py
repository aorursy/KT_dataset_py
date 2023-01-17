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
from collections import defaultdict
import nltk
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct,"")
    return x

def split_text(x):
    x = wordninja.split(x)
    return '-'.join(x)
trainSet = pd.read_csv('../input/train.csv')
testSet = pd.read_csv('../input/test.csv')
trainSet["question_text"] = trainSet["question_text"].apply(lambda x: clean_text(x))
testSet["question_text"] = testSet["question_text"].apply(lambda x: clean_text(x))
trainSet["question_text"] = trainSet["question_text"].str.lower()
testSet["question_text"] = testSet["question_text"].str.lower()
train_int,val_int = train_test_split(trainSet, test_size=0.001, random_state=2018)
train_X = train_int["question_text"].fillna("_##_").values
val_X = val_int["question_text"].fillna("_##_").values
test_X = testSet["question_text"].fillna("_##_").values
tokenizer = Tokenizer(num_words=120000)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)
trainX = np.array(train_X)
valX = np.array(val_X)
testX = np.array(test_X)
trainX = pad_sequences(trainX,maxlen = 70)
valX = pad_sequences(valX,maxlen = 70)
testX = pad_sequences(testX,maxlen = 70)
B_model = BernoulliNB(alpha=0.1,binarize = 5000)
B_model.fit(trainX,train_int.target)
predXB = B_model.predict(valX)
accuracy_score(val_int.target,predXB,normalize=True)
predXBtest = B_model.predict(testX)
result = pd.Series(predXBtest,name = "Target")
submission = pd.concat([pd.Series(testSet.qid,name = "qid"),result],axis = 1)
submission.to_csv("submit1.csv",index=False)
