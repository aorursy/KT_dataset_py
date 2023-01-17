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
import nltk
from nltk.corpus import stopwords
from ast import literal_eval
from nltk.tokenize import word_tokenize
import collections
import re
from collections import Counter
from scipy import sparse as sp
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.tsv',sep ='\t')
val = pd.read_csv('../input/validation.csv')
val.head(2)
X_train,Y_train = train['title'].values,train['tags'].values
X_test = test['title'].values
X_val,Y_val = val['title'].values,val['tags'].values
REPLACE_BY_SPACE = re.compile('[,./;\':\[\]\|]')
BAD_SYMBOLS = re.compile('[0-9#@%&]')
STOP_WORDS = set(stopwords.words('english'))
def prepare_text(text):
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE,' ',text)
    text = re.sub(BAD_SYMBOLS,'',text)
    tokens = word_tokenize(text)
    filtered_words = [w for w in tokens if not w in STOP_WORDS]
    new_text =""
    for w in filtered_words:
        if w == filtered_words[len(filtered_words)-1]:
            new_text = new_text + w
        else :
            new_text = new_text  + w + " "
    return new_text
X_train = [prepare_text(x) for  x in X_train]
selected_words = []
tags = []
for i in range (0,len(X_train)):
    selected_words = selected_words + re.findall(r'\w+',X_train[i])
    tags.append(Y_train[i])
word_count = Counter(selected_words)
tag_count = Counter(tags)
common_words = sorted(word_count.items(),reverse=True,key=lambda x:x[1])[:5000]
common_tags = sorted(tag_count.items(),reverse=True,key=lambda x:x[1])
WORD_TO_INDEX = {}
INDEX_TO_WORD= {}
for i in range(0,5000):
    WORD_TO_INDEX[common_words[i][0]] = i
    INDEX_TO_WORD[i] = common_words[i][0]
def bag_of_words(text,word_to_index,dict_size):
    y = text.split(' ')
    result = np.zeros(dict_size)
    for word in y:
        for key,index in word_to_index.items():
            if( key == word):
                result[index] = result[index] +1
    return result
        
X_train_bag = sp.vstack([sp.csr_matrix(bag_of_words(text,WORD_TO_INDEX,5000)) for text in X_train])
X_test_bag = sp.vstack([sp.csr_matrix(bag_of_words(text,WORD_TO_INDEX,5000)) for text in X_test])
print('X_train shape ', X_train_bag.shape)
print('X_test shape ', X_test_bag.shape)
mlb = MultiLabelBinarizer(classes=sorted(tag_count.keys()))
Y_train = mlb.fit_transform(Y_train) # it chnage the y_train in feature form like alll clases with 0,1 value
Y_val = mlb.fit_transform(Y_val)
model=OneVsRestClassifier(LogisticRegression()).fit(X_train,Y_train)
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_val)
print(accuracy_score(Y_val,y_pred))
