##  NOTE:  See this website for the problem under consideration here:
##        https://www.kaggle.com/c/nlp-getting-started


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#  More imports
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from spacy.lang.en import STOP_WORDS
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv').set_index('id')
train.head()
len(train)
train['target'].value_counts()
train.info()
vectorizer = TfidfVectorizer(stop_words=STOP_WORDS)
features = vectorizer.fit_transform(train['text'])
features
logr = GridSearchCV(LogisticRegression(max_iter=200), param_grid={'C': np.logspace(.01,2, 10)}, cv=5)
logr.fit(features, train['target'])
print(f"Accuracy (LogReg): {logr.score(features, train['target'])}")  # Accuracy!
print(f"Params: {logr.best_params_}")
svc = GridSearchCV(LinearSVC(max_iter=350), param_grid={'C': np.logspace(.1, 1.5, 10)}, cv=5)
svc.fit(features, train['target'])
print(svc.score(features, train['target']))
print(svc.best_params_)
nb = GridSearchCV(MultinomialNB(), param_grid={'alpha': [1, 10, 30, 50]}, cv=5)
nb.fit(features, train['target'])
nb.score(features, train['target'])
nb.best_params_
log_prob = nb.best_estimator_.feature_log_prob_

polarity = log_prob[0,:] - log_prob[1, :]
ind = np.argsort(polarity)
ind[:10]
ind_top = np.hstack((ind[:20], ind[:-21:-1]))
ind_to_token = vectorizer.get_feature_names()
top_polar_words = [ind_to_token[i] for i in ind_top]
top_polar_words[:20]
top_polar_words[20:]
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv').set_index('id')
test.head()
test_predictions = svc.predict(vectorizer.transform(test['text']))
output = pd.DataFrame(test_predictions, index=test.index, columns=['target'])
output.to_csv('/kaggle/working/predictions.csv')
len(train['location'].unique())
len(train['keyword'].unique())
train['keyword'].unique()
train['keyword'] = train['keyword'].str.replace('%20', ' ')
train['keyword'].unique()
train[train['keyword'].isna()].count()
train.loc[train['keyword'].isna(), 'keyword'] = ''
train[train['keyword'].isna()]
pipe = Pipeline([
    ('features', ColumnTransformer([
        ('k', TfidfVectorizer(stop_words=STOP_WORDS), 'keyword'),
        ('t', TfidfVectorizer(stop_words=STOP_WORDS), 'text')
    ])),
    ('svc', GridSearchCV(LinearSVC(), param_grid={'C': np.logspace(0.01,1,10)}, cv=5))
])
train.columns
X_train = train[['keyword', 'location', 'text']]
y_train = train['target']
pipe.fit(X_train, y_train)
pipe.score(X_train, y_train)
pipe.named_steps['svc'].best_params_
X_test = test
X_test['keyword'] = X_test['keyword'].str.replace('%20', ' ')
X_test.loc[X_test['keyword'].isna(), 'keyword'] = ''
test_predictions = pipe.predict(X_test)
output = pd.DataFrame(test_predictions, index=test.index, columns=['target'])
output.to_csv('/kaggle/working/new_predictions.csv')
import spacy
nlp = spacy.load('en', disable=['parser', 'dep', 'ner', 'ents'])
import re
regex = re.compile('\b*(\w+)\b*')  #  Word boundary, word, word boundary

def lemmatizer(doc):
    sent = " ".join(regex.findall(doc))
    return [w.lemma_.lower() for w in nlp(sent) if w.pos_ not in ['PRON']]
stop_words_lemmatized = [w for w in lemmatizer(" ".join(STOP_WORDS))]
lemmatizer('Yesterday, I went out and found two birds. I took some pictures and sent them to my mother.')
pipe = Pipeline([
    ('features', ColumnTransformer([
        ('k', TfidfVectorizer(stop_words=STOP_WORDS, tokenizer=lemmatizer), 'keyword'),
        ('t', TfidfVectorizer(stop_words=STOP_WORDS, tokenizer=lemmatizer), 'text')
    ])),
    ('svc', GridSearchCV(LinearSVC(), param_grid={'C': np.logspace(0.01,1,10)}, cv=5))
])
pipe.fit(X_train, y_train)
pipe.score(X_train, y_train)
test_predictions = pipe.predict(X_test)
output = pd.DataFrame(test_predictions, index=test.index, columns=['target'])
output.to_csv('/kaggle/working/predictions-Aug-17-2020.csv')
train['text']
