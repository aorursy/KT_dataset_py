# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer #calculating TF-TDF - sparse matrix
from sklearn.svm import LinearSVC #Model used for classification
from scipy.sparse import hstack # Concatinating sparse matrices
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.tsv', sep="\t")
test = pd.read_csv('../input/test.tsv', sep="\t")
sub = pd.read_csv('../input/sampleSubmission.csv', sep=",")
train.head()
test.head()
sub.head()
train_text = train['Phrase']
test_text = test['Phrase']
all_phrases = pd.concat([train_text,test_text])
word_Vectorizer=TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1,3),
    max_features=11000)

word_Vectorizer.fit(all_phrases)
train_word_features = word_Vectorizer.transform(train_text)
test_word_features = word_Vectorizer.transform(test_text)
char_Vectorizer=TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2,4),
    max_features=50000)

char_Vectorizer.fit(all_phrases)
train_char_features = word_Vectorizer.transform(train_text)
test_char_features = word_Vectorizer.transform(test_text)
train_len = [len(x) for x in train_text]
test_len = [len(x) for x in test_text]
X_train = hstack([train_char_features, train_word_features,np.array(train_len)[:,None]])
X_test = hstack([test_char_features, test_word_features,np.array(test_len)[:,None]])
y = train.Sentiment
sub['Sentiment'] = LinearSVC(dual=False).fit(X_train,y).predict(X_test)
sub.to_csv("svc.csv", index=False)
