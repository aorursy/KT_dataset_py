# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



titanic = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
titanic.head()
titanic.describe()
titanic.info()
titanic.isnull().sum()
test.isnull().sum()
%matplotlib inline

import matplotlib.pyplot as plt

titanic.hist(bins=50, figsize=(11,8))
titanic.corr()
"""

PassengerId      T

Pclass             C

Name             T

Sex                C

Age               N

SibSp             N

Parch             N

Ticket           T

Fare              N

Cabin            T

Embarked           C

"""



num_attribs = list([

    'Age',

    'SibSp',

    'Parch',

    'Fare',

    'Pclass',

])



cat_attribs = list([

    'Sex',

    'Embarked',

])



titanic.dropna(subset=cat_attribs, inplace=True)



titanic_labels = titanic['Survived'].copy()

titanic_num = titanic[num_attribs].copy()

titanic_cat = titanic[cat_attribs].copy()
titanic_num.info()
titanic_cat.info()
for column in titanic_cat.columns:

    print(titanic_cat[column].value_counts())

    print("")
from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import StandardScaler



from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values



class OneHotEncoder(BaseEstimator):

    def __init__(self):

        self.encoder = LabelBinarizer(sparse_output=True)

    def fit(self, X, y=None):

        return self.encoder.fit(X)

    def transform(self, X, y=None):

        return self.encoder.transform(X)

    def fit_transform(self, X, y=None):

        return self.encoder.fit_transform(X)

    

num_pipeline = Pipeline([

        ('selector', DataFrameSelector(num_attribs)),

        ('imputer', Imputer(strategy="median")),

        ('std_scaler', StandardScaler()),

    ])



sex_encoder = OneHotEncoder()

sex_pipeline = Pipeline([

        ('selector', DataFrameSelector(['Sex'])),

        ('encoder', sex_encoder),

    ])



emb_encoder = OneHotEncoder()

emb_pipeline = Pipeline([

        ('selector', DataFrameSelector(['Embarked'])),

        ('encoder', emb_encoder),

    ])



full_pipeline = FeatureUnion(transformer_list=[

        ("num_pipeline", num_pipeline),

        ("sex_pipeline", sex_pipeline),

        ("emb_pipeline", emb_pipeline),

    ])



titanic_prepared = full_pipeline.fit_transform(titanic)



attribs = list()

attribs.extend(num_attribs)

attribs.extend(sex_encoder.encoder.classes_)

attribs.extend(emb_encoder.encoder.classes_)

print(attribs)



titanic_prepared
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score



tree_class = DecisionTreeClassifier()

tree_scores = cross_val_score(tree_class, titanic_prepared, titanic_labels, scoring='accuracy', cv=10)

tree_scores
tree_scores.mean()
tree_scores.std()
tree_class.fit(titanic_prepared, titanic_labels)



feature_importances = tree_class.feature_importances_

sorted(zip(feature_importances, attribs), reverse=True)
from pandas.tools.plotting import scatter_matrix



attributes = ["Fare", "Survived", "Age"]

scatter_matrix(titanic[attributes], figsize=(12, 8))
attribs
titanic_prepared[:,0:4]
pd.DataFrame([[1,2],[3,4]], columns=['first','second'], index=['row1','row2'])
pd.SparseDataFrame(titanic_prepared[:,0:4])
plt.hist(titanic_prepared[:,0].todense())
titanic[titanic.Survived == 1]['Age'].hist()
titanic[titanic.Survived == 0]['Age'].hist()
titanic_test = test.copy()

titanic_test.dropna(subset=cat_attribs, inplace=True)

titanic_test_prepared = full_pipeline.transform(titanic_test)



test_results = tree_class.predict(titanic_test_prepared)

titanic_submission = titanic_test[['PassengerId']].copy()

titanic_submission['Survived'] = test_results

titanic_submission.info()
predictions_file = open("decisiontree.csv", "w")

predictions_file.write(titanic_submission.to_csv())

predictions_file.close()