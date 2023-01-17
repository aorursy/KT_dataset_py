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
import pandas as pd

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

import string

from nltk.corpus import stopwords

import nltk

from sklearn.preprocessing import StandardScaler

train = pd.read_csv("../input/nlp-getting-started/train.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

sample = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
train.head()
train.describe()
train.target.unique()
train.drop('location', axis = 1, inplace= True)

test.drop('location', axis = 1, inplace= True)

train.dropna(inplace = True)

train.drop('id', axis = 1, inplace= True)

test.drop('id', axis = 1, inplace= True)

target = train['target']

train.reset_index(inplace = True)

train.drop('index', axis = 1, inplace = True)

test.reset_index(inplace = True)

test.drop('index', axis = 1, inplace = True)
test.fillna(value = 'ablaze', axis = 1, inplace = True)
test.head()
train.head()
train.describe(include = 'all')
train["length"] = train['text'].apply(len)

test["length"] = test['text'].apply(len)
train.head()
import seaborn as sns

sns.lineplot(x = 'target', y = 'length', data = train)
sns.distplot(train[train['target']==0]['length'])
sns.distplot(train[train['target']==1]['length'])
train.groupby('target').describe(include = 'all')
train[train['length']==151]['text'].iloc[0]
train.hist('length', by = 'target')
type(train['keyword'])

train['keyword'].to_string()
# from sklearn import preprocessing

# le = preprocessing.LabelEncoder()

# le.fit(train['keyword'])

# le.transform(train['keyword'])

# train['keyword'] = le.transform(train['keyword'])

# [word for word in x.split() if word.lower() not in stopwords.words('english')]
def get_process(mess):

    x = [char for char in mess if char not in string.punctuation]

    x = ''.join(x)

    return x

train['text'] = train['text'].apply(get_process)

test['text'] = test['text'].apply(get_process)
type(train['text'])
train.head()
train['text'] = train['text'].str.lower()

test['text'] = test['text'].str.lower()
train.head()
target = train['target']

train.drop('target', axis = 1, inplace = True)
train_text = CountVectorizer(analyzer = get_process).fit(train['text'])

test_text = CountVectorizer(analyzer = get_process).fit(test['text'])

train_text_bow = train_text.transform(train['text'])

test_text_bow = test_text.transform(test['text'])
train_keyword = CountVectorizer(analyzer = get_process).fit(train['keyword'])

test_keyword = CountVectorizer(analyzer = get_process).fit(test['keyword'])

train_keyword_bow = train_keyword.transform(train['keyword'])

test_keyword_bow = test_keyword.transform(test['keyword'])
print(len(train_text.vocabulary_))
print(train_text_bow)
train_text.get_feature_names()
print('shape of sparse = ', train_text_bow.shape)
train_text_ = TfidfTransformer().fit(train_text_bow)

test_text_ = TfidfTransformer().fit(test_text_bow)

train_text_tfidf = train_text_.transform(train_text_bow)

test_text_tfidf = test_text_.transform(test_text_bow)
train_keyword_ = TfidfTransformer().fit(train_keyword_bow)

test_keyword_ = TfidfTransformer().fit(test_keyword_bow)

train_keyword_tfidf = train_keyword_.transform(train_keyword_bow)

test_keyword_tfidf = test_keyword_.transform(test_keyword_bow)
print(train_text_tfidf)
print(train_keyword_tfidf)
# from scipy.sparse import csr_matrix

# tfidf_update = tfidf.todense()

# tfidf_update1 = tfidf.toarray()
from scipy.sparse import coo_matrix, hstack
A = coo_matrix(train_text_tfidf)

B = coo_matrix(train_keyword_tfidf)

C = coo_matrix(test_text_tfidf)

D = coo_matrix(test_keyword_tfidf)
train_update = pd.DataFrame(hstack([A,B]).toarray())

test_update = pd.DataFrame(hstack([C,D]).toarray())
train_update
train_update.columns
train_update.drop(92, axis = 1, inplace = True)

train_update.drop(91, axis = 1, inplace = True)

train_update.drop(90, axis = 1, inplace = True)
test_update
solvers = ['newton-cg']

penalty = ['2']

c_values = [100, 10 , 1, 0.1, 0.01]

log_reg = LogisticRegression()

grid = dict(solver = solvers, C = c_values)

cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 1)

grid_search = GridSearchCV(estimator = log_reg, param_grid = grid, cv = cv, scoring = 'accuracy')

grid_result = grid_search.fit(train_update, target)
log = LogisticRegression()

fit = log.fit(train_update, target)
grid_result
print("best %f using %s"%(grid_result.best_score_, grid_result.best_params_))
prediction = grid_result.predict(test_update)
# log = LogisticRegression().fit(tfidf, target)

# predict_log = log.predict(tfidf_test)
# submission = pd.DataFrame()

# submission['target'] = prediction

# submission.reset_index(inplace = True)

# submission = submission.rename(columns = {'index':'id'}) 

output = pd.DataFrame({'id': sample.id, 'target': prediction})

output
output.to_csv('submission.csv', index=False)