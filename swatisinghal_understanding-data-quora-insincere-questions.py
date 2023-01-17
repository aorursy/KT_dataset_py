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
#Reading train and test data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print("Train data shape: ",train.shape)
print("Test data shape: ",test.shape)
#Class count
count_class_0, count_class_1 = train.target.value_counts()
print("Count of class 0", count_class_0)
print("Count of class 1", count_class_1)

class_0 = train[train['target'] == 0]
class_1 = train[train['target'] == 1]
print(class_0.shape)
print(class_1.shape)
train.head()
test.head()
import nltk
from nltk.corpus import stopwords
import re
#Remove bad symbols and stopwords from test and train data
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower()   # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(" ", text)     # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub("", text)     # delete symbols which are in BAD_SYMBOLS_RE from text
    
    
    resultwords = [word for word in text.split() if word not in STOPWORDS]  # delete stopwords from text
    text = ' '.join(resultwords)
    
    return text

X_train_class_0 = class_0['question_text'].values
X_train_class_1 = class_1['question_text'].values
X_train_class_0 = [text_prepare(x) for x in X_train_class_0]
X_train_class_1 = [text_prepare(x) for x in X_train_class_1]
#Word counts of n-grams of both the classes
from nltk.util import ngrams
from collections import Counter
words_counts_class_0_unigram = Counter()
words_counts_class_1_unigram = Counter()
words_counts_class_0_bigram = Counter()
words_counts_class_1_bigram = Counter()
words_counts_class_0_trigram = Counter()
words_counts_class_1_trigram = Counter()

for sentence in X_train_class_0:
  token = [word for word in sentence.split()]
  words_counts_class_0_unigram.update(x for x in ngrams(token, 1))
  words_counts_class_0_bigram.update(x for x in ngrams(token, 2))
  words_counts_class_0_trigram.update(x for x in ngrams(token, 3))

for sentence in X_train_class_1:
  token = [word for word in sentence.split()]
  words_counts_class_1_unigram.update(x for x in ngrams(token, 1))
  words_counts_class_1_bigram.update(x for x in ngrams(token, 2))
  words_counts_class_1_trigram.update(x for x in ngrams(token, 3))

words_counts_class_0_unigram.most_common(10)
words_counts_class_1_unigram.most_common(10)
words_counts_class_0_bigram.most_common(10)
words_counts_class_1_bigram.most_common(10)
words_counts_class_0_trigram.most_common(10)
words_counts_class_1_trigram.most_common(10)
print('Average character length of Sincere questions in train is {0:.0f}.'.format(np.mean(class_0['question_text'].apply(lambda x: len(x)))))
print('Average character length of Insincere questions in train is {0:.0f}.'.format(np.mean(class_1['question_text'].apply(lambda x: len(x)))))
print('Average word length of Sincere questions in train is {0:.0f}.'.format(np.mean(class_0['question_text'].apply(lambda x: len(x.split())))))
print('Average word length of Insincere questions in train is {0:.0f}.'.format(np.mean(class_1['question_text'].apply(lambda x: len(x.split())))))
print('Max word length of Sincere questions in train is {0:.0f}.'.format(np.max(class_0['question_text'].apply(lambda x: len(x.split())))))
print('Max word length of Insincere questions in train is {0:.0f}.'.format(np.max(class_1['question_text'].apply(lambda x: len(x.split())))))
