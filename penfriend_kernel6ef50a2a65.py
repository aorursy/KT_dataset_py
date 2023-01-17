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
train = pd.read_csv('../input/train.csv').fillna(' ')
train_text = train.question_text



test = pd.read_csv('../input/test.csv').fillna(' ')

test_text = test.question_text
all_text = pd.concat([train_text, test_text])
#print(train_df.columns)
#print(train_df.question_text)
#print(train_df.shape)
#for x in train_df:
#    print(x)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
#def createDTM(messages):
#    vect = TfidfVectorizer()
#    dtm = vect.fit_transform(messages) # create DTM
    
    # create pandas dataframe of DTM
#    return pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names()) 
#messages = train_text.question_text
#createDTM(messages)
class_names = ['target']
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
scores = []
submission = pd.DataFrame.from_dict({'id': test['qid']})
for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(C=0.1, solver='sag')

    cv_score = np.mean(cross_val_score(classifier, train_word_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_word_features, train_target)
    submission[class_name] = classifier.predict_proba(test_word_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('submission.csv', index=False)
print(word_vectorizer.vocabulary_)