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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()
classes = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']
train_text = train.comment_text
test_text = test.comment_text
all_text = pd.concat([train_text,test_text])
all_text.head()
word_vect = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vect.fit(all_text)
train_word_feats = word_vect.transform(train_text)
test_word_feats = word_vect.transform(test_text)
char_vect = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=60000)
char_vect.fit(all_text)
train_char_feats = char_vect.transform(train_text)
test_char_feats = char_vect.transform(test_text)

train_feats = hstack([train_word_feats,train_char_feats])
test_feats = hstack([test_word_feats, test_char_feats])
scores = []
submission = pd.DataFrame.from_dict({'id' : test['id']})
for class_name in classes :
    train_target = train[class_name]
    classifier = LogisticRegression(C=0.1, solver='sag')
    
    cv_score = np.mean(cross_val_score(classifier, train_feats, train_target,cv=3,scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))
    
    classifier.fit(train_feats,train_target)
    submission[class_name] = classifier.predict_proba(test_feats)[:,1]
    
print("Total CV score is {}".format(np.mean(scores)))


scores
#submission.to_csv('submission.csv')