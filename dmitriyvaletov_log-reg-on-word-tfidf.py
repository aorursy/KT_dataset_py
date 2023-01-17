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
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

from scipy.sparse import hstack
train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')
def prepear_data(train, test):
    
#     yes_df = train[train['target'] == 1]
#     no_df = train[train['target'] == 0]
#     train_df = yes_df.append(yes_df).append(no_df.loc[:yes_df.shape[0]])
    train_df = train
    
#     print('target: positive: {} negative: {}'.format(yes_df.shape[0], no_df.shape[0]))
    
    
    train_texts = train_df['question_text']
    test_texts = test['question_text']
    all_texts = pd.concat([train_texts, test_texts])

    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 2),
        max_features=10000)

    word_vectorizer.fit(all_texts)

    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        stop_words='english',
        ngram_range=(2, 6),
        max_features=10000)

    char_vectorizer.fit(all_texts)

    
    train_word_features = word_vectorizer.transform(train_texts)
    test_word_features = word_vectorizer.transform(test_texts)
    
    train_char_features = char_vectorizer.transform(train_texts)
    test_char_features = char_vectorizer.transform(test_texts)

    train_features = hstack([train_char_features, train_word_features])
    test_features = hstack([test_char_features, test_word_features])
    
    train_target = train_df['target']
    
    return train_features, train_target, test_features
#     return train_word_features, train_target, test_word_features
train_features, train_target, test_features = prepear_data(train, test)
classifier = LogisticRegression(C=0.1, solver='sag')
chosen_scoring = 'recall' # 'roc_auc', 'accuracy', 'recall'
cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=8, scoring=chosen_scoring))
print('CV score ({}) for target 1 is {}'.format(chosen_scoring, cv_score))
classifier.fit(train_features, train_target)
predicted_targets = classifier.predict(train_features)
print(classification_report(train_target, predicted_targets))
submission = pd.DataFrame.from_dict({'qid': test['qid']})
submission['prediction'] = classifier.predict(test_features)
submission.to_csv('submission.csv', index=False)
submission.head(5)
print(submission[submission['prediction']>0.5].shape, submission[submission['prediction'] < 0.5].shape)