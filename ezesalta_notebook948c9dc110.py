# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (15, 5)

import xgboost
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn_pandas import DataFrameMapper

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Load the data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv',)
df_train.head()
# Look for null values
null_value_stats = df_train.isnull().sum(axis=0)
print(null_value_stats[null_value_stats != 0])

# Fill null values
df_train = df_train[~df_train.Embarked.isnull()]
df_train.Age.fillna(-999, inplace=True)
df_train.Cabin.fillna('A111', inplace=True)
df_train.shape
def total(series):
    return df_train[series.name].count()

def proportion(series):
    return series.count() / df_train[series.name].count()

def title_name(series):
    def get_title(name):
        return [w for w in name.split() if '.' in w][0]
    return np.array([get_title(name) for name in series])
df_train.groupby('Sex').Survived.agg(['count', total, proportion])
df_train.groupby('Pclass').Survived.agg(['count', total, proportion])
df_train.groupby(['Pclass', 'Sex']).Survived.agg(['count', total, proportion])
df_train[(df_train.Survived == 1) & (df_train.Age > 0)].hist(['Age', 'Pclass'])
df_train[(df_train.Survived == 0) & (df_train.Age > 0)].hist(['Age', 'Pclass'])

df_title = pd.DataFrame({'Title': title_name(df_train.Name)})
df_train_title = pd.merge(df_train, df_title, left_index=True, right_index=True)
df_train_title.groupby('Title').Survived.agg(['count']).sort_values('count', ascending=False)
# Features

class Feature:
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        return X.reshape(-1, 1)

class Title(Feature):
    
    def transform(self, X):
        return np.array([self.get_title(x) for x in X])
    
    def get_title(self, x):
        return [w for w in x.split() if '.' in w][0]

%%time
mapper = DataFrameMapper([
    ('Pclass', None),
    ('Sex', LabelBinarizer()),
    ('Age', None),
    ('Fare', None),
    ('Embarked', LabelBinarizer()),
    ('Name', make_pipeline(
        Title(),
        CountVectorizer(),
    )),
])

model = make_pipeline(
    make_union(
        mapper,
    ),
    # LogisticRegression(),
    CatBoostClassifier(logging_level='Silent')
)

X = df_train.drop('Survived', axis=1)
y = df_train.Survived

y_pred = cross_val_predict(model, X, y, cv=3)
print('Accuracy:', accuracy_score(y, y_pred))
predictor = model.fit(df_train, df_train.Survived)
idx = df_test.PassengerId
y_pred = predictor.predict(df_test).astype('int')
df_submission = pd.DataFrame({'PassengerId': idx, 'Survived': y_pred})
df_submission.set_index('PassengerId', inplace=True)
df_submission
df_submission.to_csv('/tmp/submission.csv')

