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
df = pd.read_csv('../input/train.csv')
df.head()
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df[df['insult']==1]['comment_text'].values
train = pd.read_csv('../input/train.csv').fillna(' ')

test = pd.read_csv('../input/test.csv').fillna(' ')

train_text = train['comment_text']

test_text = test['comment_text']

all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1, 1),

    max_features=10000)

word_vectorizer.fit(all_text)

train_word_features = word_vectorizer.transform(train_text)

test_word_features = word_vectorizer.transform(test_text)

train_word_features.shape
test_word_features.shape
char_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='char',

    stop_words='english',

    ngram_range=(2, 6),

    max_features=50000)

char_vectorizer.fit(all_text)

train_char_features = char_vectorizer.transform(train_text)

test_char_features = char_vectorizer.transform(test_text)
train_char_features.shape
test_char_features.shape
train_features = hstack([train_char_features, train_word_features])

test_features = hstack([test_char_features, test_word_features])

scores = []

submission = pd.DataFrame.from_dict({'id': test['id']})

for class_name in class_names:

    train_target = train[class_name]

    classifier = LogisticRegression(C=0.1, solver='sag')



    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))

    scores.append(cv_score)

    print('CV score for class {} is {}'.format(class_name, cv_score))



    classifier.fit(train_features, train_target)

    submission[class_name] = classifier.predict_proba(test_features)[:, 1]



print('Total CV score is {}'.format(np.mean(scores)))



submission.to_csv('submission.csv', index=False)