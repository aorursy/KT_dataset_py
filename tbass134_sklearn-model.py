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
import pandas as pd
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_df.head()
test_df.head()
train_df.keyword.value_counts()
#base model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer
from sklearn_pandas import CategoricalImputer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV


vect = CountVectorizer()
nb = MultinomialNB()
X = train_df[['text']]
y = train_df.target

pipe = Pipeline([
    ('columntransformer', ColumnTransformer([
         ('countvectorizer', vect, 'text')],
            remainder='drop')),
    ('multinomialnb', nb)
])
f1 = cross_val_score(pipe, X, y, cv=5, scoring="f1").mean()
acc = cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()
f1, acc

params = {}
params["columntransformer__countvectorizer__lowercase"] = [True, False]
params["columntransformer__countvectorizer__stop_words"] = [None, "english"]
params["columntransformer__countvectorizer__ngram_range"] = [(1,1), (1, 2), (1, 3)]
params["multinomialnb__alpha"] = [0.1, 1, 10, 50, 100, 200]

from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(pipe, params)
f1 = cross_val_score(gs, X, y, cv=5, scoring="f1").mean()
acc = cross_val_score(gs, X, y, cv=5, scoring="accuracy").mean()
f1, acc
vect = TfidfVectorizer()
X = train_df[['text']]
y = train_df.target

pipe = Pipeline([
    ('columntransformer', ColumnTransformer([
         ('countvectorizer', vect, 'text')],
            remainder='drop')),
    ('multinomialnb', nb)
])
params = {}
params["columntransformer__countvectorizer__lowercase"] = [True, False]
params["columntransformer__countvectorizer__stop_words"] = [None, "english"]
params["columntransformer__countvectorizer__ngram_range"] = [(1,1), (1, 2), (1, 3)]
params["columntransformer__countvectorizer__max_df"] = [0.1, 0.5, 0.8, 1]
params["multinomialnb__alpha"] = [0.1, 1, 10, 50, 100, 200]

gs = RandomizedSearchCV(pipe, params)
f1 = cross_val_score(gs, X, y, cv=5, scoring="f1").mean()
acc = cross_val_score(gs, X, y, cv=5, scoring="accuracy").mean()
f1, acc
# use keyword and location
vect = TfidfVectorizer()
nb = MultinomialNB()

X = train_df[['keyword', 'location', 'text']]
y = train_df.target


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


text_transformer = Pipeline([
    ('columntransformer', ColumnTransformer([
         ('countvectorizer', vect, 'text')],
            remainder='drop'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('cat_transformer', categorical_transformer, ['keyword', 'location']),
        ('text_transformer', text_transformer, ['text'])])


pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', nb)])

f1 = cross_val_score(pipe, X, y, cv=5, scoring="f1").mean()
acc = cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()
f1, acc
params = {}
params["preprocessor__text_transformer__columntransformer__countvectorizer__stop_words"] = [None, "english"]
params["preprocessor__text_transformer__columntransformer__countvectorizer__ngram_range"] = [(1,1), (1, 2), (1, 3)]
params["preprocessor__text_transformer__columntransformer__countvectorizer__max_df"] = [0.1, 0.5, 0.8, 1]
params["classifier__alpha"] = [0.1, 1, 10, 50, 100, 200]
gs = RandomizedSearchCV(pipe, params)
f1 = cross_val_score(gs, X, y, cv=5, scoring="f1").mean()
acc = cross_val_score(gs, X, y, cv=5, scoring="accuracy").mean()
f1, acc
# Ensembling models

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer


vect = TfidfVectorizer()

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

X = train_df[['text']]
y = train_df.target

pipe1 = Pipeline([
    ('columntransformer', ColumnTransformer([
         ('countvectorizer', vect, 'text')],
            remainder='drop')),
    ('clf', clf1)
])

pipe2 = Pipeline([
    ('columntransformer', ColumnTransformer([
         ('countvectorizer', vect, 'text')],
            remainder='drop')),
    ('clf', clf2)
])

pipe3 = Pipeline([
    ('columntransformer', ColumnTransformer([
         ('countvectorizer', vect, 'text')],
            remainder='drop')),
    ('functransformer',FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
    ('clf', clf3)
])


eclf = VotingClassifier(
    estimators=[('lr', pipe1), ('rf', pipe2), ('gnb', pipe3)],
    voting='hard')

for clf, label in zip([pipe1, pipe2, pipe3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

eclf.fit(X, y)
#submission
# pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
X_new = test_df[['text']]
pd.DataFrame({'id':test_df.id, 'target':eclf.predict(X_new)}).set_index('id').to_csv('sub1.csv')
#can we train a model on just location and keyword
!kaggle competitions submit -c nlp-getting-started -f /output/kaggle/working/sub1.csv -m "sklearn model - voting classifier"
