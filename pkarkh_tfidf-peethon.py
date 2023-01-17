# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import json
from sklearn.cross_validation import train_test_split
def read_dataset(path):
    return json.load(open(path)) 
train_data = read_dataset('../input/train.json')
test_data = read_dataset('../input/test.json')
def generate_text(data):
    text_data = [" ".join(doc['ingredients']).lower() for doc in data]
    return text_data 
train_text = generate_text(train_data)
test_text = generate_text(test_data)
Y = [doc['cuisine'] for doc in train_data]
_Id = [doc['id'] for doc in test_data]
from sklearn.feature_extraction.text import TfidfVectorizer
def remove_spec(word):
    spec = ['-', '®', ' ']
    word = word.lower()
    word = list(word)
    for char in word:
        if char in spec:
            word.remove(char)
    return ''.join(word)
def n_gram(word):
    word = remove_spec(word)
    if (len(word) < 3):
        return word
    word = word.lower()
    n_length = 3
    arr = []
    if (len(word) % 2 == 0):
        word_length = len(word)
    else:
        word_length = len(word) - 1
    for i in range(word_length):
        arr.append(word[i:i + n_length])
    return arr

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, tokenizer=n_gram, ngram_range=(2,2))
X = tfidf_vectorizer.fit_transform(train_text)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
lg = LogisticRegression(
    penalty='l2',
    C=10, 
    n_jobs=-1, verbose=1, 
    solver='sag', multi_class='multinomial',
    max_iter=300
)
lg.fit(X_train, Y_train)
vocab = tfidf_vectorizer.vocabulary_.items()
vocab = sorted(list(vocab), key=lambda x: x[1])
vocab_words, vocab_index = zip(*vocab)
vocab_words = np.array(vocab_words)
for label in range(20):
    _class_coef = lg.coef_[label]
    print('Class', label, 'words increasing the probability of a class:')
    print(list(vocab_words[ (-_class_coef).argsort()][:100]))
    print()
    print('Class', label,  'words decreasing the probability of a class:')
    print(list(vocab_words[ (_class_coef).argsort()][:100]))
    print('-'*80)
Y_pred = lg.predict(X_test)
print(classification_report(Y_test, Y_pred, digits=6))
X_ = tfidf_vectorizer.transform(test_text)
Y_target = lg.predict(X_)
with open('submit.csv', 'w') as f:
    f.write('id,cuisine\n')
    for _id, y  in zip(_Id, Y_target):
        f.write('%s,%s\n' % (_id, y))
