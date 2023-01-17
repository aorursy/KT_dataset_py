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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_train.head()
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_test.head()
df_train.shape, df_test.shape
print(df_train.isnull().sum())

print('**********************************************************')

print(df_test.isnull().sum())
df_train.drop(['Cabin'], axis = 1, inplace = True)

df_test.drop(['Cabin'], axis = 1, inplace = True)
df_train['Fam_size'] = df_train['SibSp'] + df_train['Parch'] + 1

df_test['Fam_size'] = df_test['SibSp'] + df_test['Parch'] + 1
df_train['Fam_type'] = pd.cut(df_train.Fam_size, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])

df_test['Fam_type'] = pd.cut(df_test.Fam_size, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])
df_train['Title'] = df_train['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

df_test['Title'] = df_test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df_train.head()
df_train['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)

df_test['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)



df_train['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)

df_test['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)
df_train_copy = df_train.copy()

df_test_copy = df_test.copy()
df_train_copy.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)

df_test_copy.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace =True)
df_train_copy.head()
y = df_train['Survived']

features = ['Pclass', 'Sex', 'Fare', 'Title', 'Embarked', 'Fam_type']

X = df_train[features]

X.head()
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

numerical_cols = ['Fare']

categorical_cols = ['Pclass', 'Sex', 'Title', 'Embarked', 'Fam_type']



numerical_transformer = SimpleImputer(strategy='median')



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder())

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
from sklearn.ensemble import RandomForestClassifier



titanic_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', RandomForestClassifier(random_state=0, n_estimators=500, max_depth=5))])



titanic_pipeline.fit(X,y)



print('Cross validation score: {:.3f}'.format(cross_val_score(titanic_pipeline, X, y, cv=10).mean()))
X_test = df_test[features]

X_test.head()
y_pred_random_forest = titanic_pipeline.predict(X_test)
from sklearn.linear_model import LogisticRegression

titanic_pipeline1 = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', LogisticRegression(random_state=0))])



titanic_pipeline1.fit(X,y)



print('Cross validation score: {:.3f}'.format(cross_val_score(titanic_pipeline1, X, y, cv=10).mean()))
y_pred_lr = titanic_pipeline1.predict(X_test)
from sklearn.svm import SVC

titanic_pipeline2 = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', SVC(kernel = 'linear', random_state=0))])



titanic_pipeline2.fit(X,y)



print('Cross validation score: {:.3f}'.format(cross_val_score(titanic_pipeline2, X, y, cv=10).mean()))
y_pred_svm = titanic_pipeline2.predict(X_test)
from sklearn.neighbors import KNeighborsClassifier

titanic_pipeline3 = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2))])



titanic_pipeline3.fit(X,y)



print('Cross validation score: {:.3f}'.format(cross_val_score(titanic_pipeline3, X, y, cv=10).mean()))
y_pred_knn = titanic_pipeline3.predict(X_test)
from sklearn.naive_bayes import GaussianNB

titanic_pipeline4 = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', GaussianNB())])



titanic_pipeline4.fit(X,y)



print('Cross validation score: {:.3f}'.format(cross_val_score(titanic_pipeline4, X, y, cv=10).mean()))
y_pred_gaussian = titanic_pipeline4.predict(X_test)
from sklearn.tree import DecisionTreeClassifier

titanic_pipeline5 = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', DecisionTreeClassifier(criterion = 'entropy'))])



titanic_pipeline5.fit(X,y)



print('Cross validation score: {:.3f}'.format(cross_val_score(titanic_pipeline5, X, y, cv=10).mean()))
y_pred_decision_tree = titanic_pipeline5.predict(X_test)
from sklearn.linear_model import SGDClassifier

titanic_pipeline6 = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', SGDClassifier())])



titanic_pipeline6.fit(X,y)



print('Cross validation score: {:.3f}'.format(cross_val_score(titanic_pipeline6, X, y, cv=10).mean()))
y_pred_sgd = titanic_pipeline6.predict(X_test)
from sklearn.svm import LinearSVC

titanic_pipeline7 = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', LinearSVC())])



titanic_pipeline7.fit(X,y)



print('Cross validation score: {:.3f}'.format(cross_val_score(titanic_pipeline7, X, y, cv=10).mean()))
y_pred_svc = titanic_pipeline7.predict(X_test)
from sklearn.linear_model import Perceptron

titanic_pipeline8 = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', Perceptron(tol = 1e-3, random_state = 0))])



titanic_pipeline8.fit(X,y)



print('Cross validation score: {:.3f}'.format(cross_val_score(titanic_pipeline8, X, y, cv=10).mean()))
y_pred_perceptron = titanic_pipeline8.predict(X_test)
from xgboost import XGBClassifier

titanic_pipeline9 = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', XGBClassifier())])



titanic_pipeline9.fit(X,y)



print('Cross validation score: {:.3f}'.format(cross_val_score(titanic_pipeline9, X, y, cv=10).mean()))
y_pred_xgb = titanic_pipeline9.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": y_pred_random_forest

    })