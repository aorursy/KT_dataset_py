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



# Further imports (RAM)

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt



# Scikit-learn imports

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, TransformerMixin

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
train_df = pd.read_csv('/kaggle/input/titanic/train.csv').set_index('PassengerId')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv').set_index('PassengerId')
train_df.head()
train_df.describe()
train_df.info()
test_df.info()
train_df['Survived'].value_counts()
train_df.hist(column='Survived', by=['Sex'], sharey=True);
pd.crosstab(train_df['Survived'], train_df['Sex'], margins=True)
train_df[['Sex', 'Survived']].groupby('Sex').mean()
train_df['Embarked'].unique()
train_df['Embarked'].value_counts()
pd.crosstab(train_df['Survived'], train_df['Embarked'], margins=True)
pd.crosstab(train_df['Survived'], train_df['Pclass'], margins=True)
train_df[['Survived', 'Pclass']].groupby('Pclass').mean()
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2, aspect=1.4)

grid.map(plt.hist, 'Age', alpha=0.5, bins=20)

grid.add_legend();
train_df[['Sex', 'Pclass', 'Age']].groupby(['Sex', 'Pclass']).median().to_dict()['Age']
class AgeImputer(BaseEstimator, TransformerMixin):

    """ Assumes input is a DataFrame. Output will also be in the form of a DataFrame."""

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        self.ages = X[['Sex', 'Pclass', 'Age']].groupby(['Sex', 'Pclass']).median().to_dict()['Age']

        return self

    

    def transform(self, X):

        X = X.copy()

        for sex in ['female', 'male']:

            for pclass in range(1,4):

                X.loc[ (X['Age'].isnull()) 

                      & (X['Sex'] == sex) 

                      & (X['Pclass'] == pclass),'Age'] = self.ages[(sex, pclass)]

        return X
age_imputer = AgeImputer()

new_train_df = age_imputer.fit_transform(train_df)
new_train_df[new_train_df['Age'].isnull()]
class GenderClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

    

    def predict(self, X):

        return [1 if row.Sex == 'female' else 0 for row in X.itertuples()]
gender = GenderClassifier()

gender.predict(new_train_df)[:10]
new_train_df.head()
gender.score(new_train_df, new_train_df['Survived'])
gender_output = pd.DataFrame(gender.predict(test_df), index=test_df.index, columns=['Survived'])
gender_output.head()
gender_output.to_csv('/kaggle/working/gender.csv')
X_train = train_df.drop('Survived', axis=1)

y_train = train_df['Survived']
X_train['Cabin'].value_counts()
features = Pipeline([

    ('age', AgeImputer()),

    ('impute', ColumnTransformer([

        ('fare', SimpleImputer(strategy='median'), ['Fare']),

        ('embarked', SimpleImputer(strategy='most_frequent'), ['Embarked']),

        ('keep', 'passthrough', ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])

    ])),

    ('onehot', ColumnTransformer([

        ('code', OneHotEncoder(), [1, 2, 3])

    ], remainder='passthrough'))

])
features.fit_transform(X_train)[:3]
est = Pipeline([

    ('features', features),

    ('rfr', GridSearchCV(RandomForestClassifier(), param_grid={'max_depth': [10, 12, 15, 20, 25],

                                                              'min_samples_leaf': [5, 10],

                                                              'min_samples_split': [3, 5]},

                        cv=5, verbose=1, n_jobs=2))

])
est.fit(X_train, y_train)
est.score(X_train, y_train)
est.named_steps['rfr'].best_params_
rfr_output =pd.DataFrame(est.predict(test_df), index=test_df.index, columns=['Survived'])
rfr_output.head()
rfr_output.to_csv('/kaggle/working/rfr.csv')