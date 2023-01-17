import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import tarfile

import urllib



%matplotlib inline

%reload_ext autoreload

%autoreload 2

np.random.seed(42)
print(os.listdir("../input"))
from sklearn.model_selection import train_test_split



df_train = pd.read_csv('../input/titanic/train.csv', header=0, sep=',', quotechar='"')

df_test = pd.read_csv('../input/titanic/test.csv', header=0, sep=',', quotechar='"')

df_x_train = df_train.drop('Survived', axis=1)

y_train = df_train['Survived']
df_expl = df_x_train.copy()
df_expl.head()
df_expl.head(10)
df_expl.info()
# To print out the classes for nominal variable.



cols_cat = [ 'Pclass', 'Sex', 'Embarked']

for _ in cols_cat:

  print('-' * 40)

  print(f'{_} : {df_expl[_].value_counts()}')
# https://www.kaggle.com/c/titanic/data

# sibsp refers to: # of siblings / spouses aboard the Titanic	

# parch	referts to: # of parents / children aboard the Titanic



pd.set_option('display.max_colwidth', -1)

cols_expl_title = ['Name', 'Age', 'SibSp', 'Parch']

df_expl[cols_expl_title].head(20)
titles = [i.split(',')[1].split('.')[0].strip() for i in df_expl['Name']]
df_expl['Title'] = pd.Series(titles)

df_expl['Title'].value_counts()
titles_common = ['Mr', 'Miss', 'Mrs', 'Master']

df_expl.loc[~df_expl['Title'].isin(titles_common)]
# uncommon title will be assign to the value 'rare'



df_expl.loc[~df_expl['Title'].isin(titles_common), 'Title'] = 'rare'

df_expl['Title'].value_counts()
df_expl.head()
df_expl[df_expl['Ticket']=='PC 17611']
df_expl[df_expl['Fare']>130].sort_values(by='Fare')
# sibsp refers to: # of siblings / spouses aboard the Titanic	

# parch	referts to: # of parents / children aboard the Titanic



df_expl[df_expl['Ticket']=='PC 17757']
df_expl[df_expl['Ticket']=='PC 17582']
df_expl[df_expl['Ticket']=='PC 17611']
df_expl[df_expl['Name'].str.contains('Dennick')]

df_expl['Title'].value_counts()
df_expl.groupby(['Sex','Title']).describe()
# X_validation, X_test, y_validation, y_test 

# ColumnTransformer mueller chapter 4



from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.pipeline import FeatureUnion

from sklearn.model_selection import GridSearchCV





class SelectColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns_names):

        self.columns_names = columns_names

    def fit(self, X, y=None):

    # y=None specifies we don't want to affect the label set

        return self

    def transform(self, X):

        return X[self.columns_names]



class MostFrequentValue(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                        index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)





num_pipeline = Pipeline([

        ("select_numeric", SelectColumns(["Age", "SibSp", "Parch", "Fare"])),

        ("imputer", SimpleImputer(strategy="median")),

        ("scaler", MinMaxScaler()),

    ])



cat_pipeline = Pipeline([

        ("select_cat", SelectColumns(["Pclass", "Sex", "Embarked"])),

        ("imputer", MostFrequentValue()),

        ("cat_encoder", OneHotEncoder(sparse=False)),

    ])



preprocess_pipeline = FeatureUnion(transformer_list=[

        ("num_pipeline", num_pipeline),

        ("cat_pipeline", cat_pipeline),   

    ])
df_expl.head()
X_train = preprocess_pipeline.fit_transform(df_x_train)

X_train
from sklearn.svm import SVC



svm_clf = SVC(gamma="auto")

svm_clf.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score



svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)

svm_scores.mean()
from sklearn.ensemble import RandomForestClassifier



# n_estimators refers to the number of trees in the forest.

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)

forest_scores.mean()
forest_clf
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

from sklearn.ensemble import RandomForestRegressor





param_distribs = {

        'n_estimators': randint(low=1, high=200),

        'max_features': randint(low=1, high=8),

    }



forest_reg = RandomForestRegressor(random_state=42)

rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,

                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)

rnd_search.fit(X_train, y_train)
cvres = rnd_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
rnd_search.best_params_
# n_estimators refers to the number of trees in the forest.

forest_clf = RandomForestClassifier(n_estimators=122, max_features=7, random_state=42)

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)

forest_scores.mean()
X_test = preprocess_pipeline.fit_transform(df_test)
forest_clf.fit(X_train, y_train)
predictions = forest_clf.predict(X_test)
submission = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':predictions})

submission.head()

filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)