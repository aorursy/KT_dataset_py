# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head()
train.describe()
var = 'Fare'

neg = train[var][train['Survived'] == 0].dropna()
pos = train[var][train['Survived'] == 1].dropna()

plt.figure(figsize=(10,8))
plt.title('Comparison of {0} distribution when controlled by Survived'.format(var) )
sns.distplot(neg, color='r', label='Died', kde=False, norm_hist=True)#, bins=range(0, 81, 5))
sns.distplot(pos, color='b', label='Survived', kde=False, norm_hist=True)#, bins=range(0, 81, 5))
plt.legend(loc='best')
plt.show()
var = 'Cabin'
print('{0} class distribution'.format(var))
print('% null values:', 100 * train[var].isnull().sum() / len(train))

x = train[~train[var].isnull()]
for i in sorted(x[var].unique()):
    data = x[x[var] == i]
    print('class = {0}, total = {1} %, survived = {2} %, died = {3} %'.format(i, 100 * len(data)/ len(x), 100 * len(data[data['Survived'] == 1]) / len(data), 100 * len(data[data['Survived'] == 0]) / len(data)))
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class Binarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.binarizer = LabelBinarizer()
    
    def fit(self, X, y=None):
        self.binarizer.fit(X)
        return self

    def transform(self, X):
        return self.binarizer.transform(X)
    
features = ['SibSp', 'Parch', 'Fare', 'Pclass', 'Embarked', 'Sex']
X_train = train[features].dropna()
y_train = train['Survived'].loc[X_train.index]

feature_engineering_pipeline = ColumnTransformer([
    ('standardizer', StandardScaler(), ['Fare', 'SibSp', 'Parch']),
    ('one-hot', OneHotEncoder(), ['Pclass', 'Embarked']),
    ('binarizer', Binarizer(), ['Sex'])
], remainder='drop')

pipeline = Pipeline([
    ('feature_engineering', feature_engineering_pipeline),
    ('model', RandomForestClassifier())
])

param_grid = {
    'model__n_estimators': [1000],
    'model__max_depth': [4, 8, 16]
    #'model__C': [0.1, 1, 10, 1e2, 1e3],
    #'model__kernel': ['rbf']
    #'model__gamma': [1, 0.1, 0.01, 0.001],
}

clf_cv = GridSearchCV(pipeline, param_grid, 
                      iid=False, 
                      cv=5,
                      scoring='accuracy',
                      n_jobs=-1,
                      return_train_score=False)

clf_cv.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % clf_cv.best_score_)
print(clf_cv.best_params_)
test = pd.read_csv('../input/test.csv')
X_test = test[features].dropna()
y_test = clf_cv.predict(X_test)
output = pd.Series(y_test, index=X_test.index)
