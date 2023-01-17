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
!pip install sklearn_crfsuite



import sklearn_crfsuite

from sklearn_crfsuite import scorers

from sklearn_crfsuite import metrics



from itertools import chain



import nltk

import sklearn

import scipy.stats

from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV
def word2features(sent, i):

    word = sent[i][0]

    postag = sent[i][1]



    features = {

        'bias': 1.0,

        'word.lower()': word.lower(),

        'word[-3:]': word[-3:],

        'word[-2:]': word[-2:],

        'word.isupper()': word.isupper(),

        'word.istitle()': word.istitle(),

        'word.isdigit()': word.isdigit(),

        'postag': postag,

        'postag[:2]': postag[:2],

    }

    if i > 0:

        word1 = sent[i-1][0]

        postag1 = sent[i-1][1]

        features.update({

            '-1:word.lower()': word1.lower(),

            '-1:word.istitle()': word1.istitle(),

            '-1:word.isupper()': word1.isupper(),

            '-1:postag': postag1,

            '-1:postag[:2]': postag1[:2],

        })

    else:

        features['BOS'] = True



    if i < len(sent)-1:

        word1 = sent[i+1][0]

        postag1 = sent[i+1][1]

        features.update({

            '+1:word.lower()': word1.lower(),

            '+1:word.istitle()': word1.istitle(),

            '+1:word.isupper()': word1.isupper(),

            '+1:postag': postag1,

            '+1:postag[:2]': postag1[:2],

        })

    else:

        features['EOS'] = True



    return features





def sent2features(sent):

    return [word2features(sent, i) for i in range(len(sent))]



def sent2labels(sent):

    return [label for token, postag, label in sent]



def sent2tokens(sent):

    return [token for token, postag, label in sent]
nltk.corpus.conll2002.fileids()



train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))

test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))



sent = ""

pos = ""

tag = ""

for token in train_sents[0]:

    sent += str(token[0])+" "

    pos += str(token[1])+" "

    tag += str(token[2])+" "



print(sent)

print(pos)

print(tag)
sent2features(train_sents[0])[0]
X_train = [sent2features(s) for s in train_sents]

y_train = [sent2labels(s) for s in train_sents]



X_test = [sent2features(s) for s in test_sents]

y_test = [sent2labels(s) for s in test_sents]
crf = sklearn_crfsuite.CRF(

    algorithm='lbfgs',

    c1=0.1,

    c2=0.1,

    max_iterations=100,

    all_possible_transitions=True

)

crf.fit(X_train, y_train)
print("Sent:", [aug_w[0] for aug_w in train_sents[0]])

sent_f = sent2features(train_sents[0])

print("Sent f shape:", len(sent_f), ", feats:", sent_f[0])

pred = crf.predict([sent_f])

print("Predicted:", pred)
labels = list(crf.classes_)

labels.remove('O')

labels
y_pred = crf.predict(X_test)

metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)