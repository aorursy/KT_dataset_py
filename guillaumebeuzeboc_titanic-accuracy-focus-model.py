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



train_set = pd.read_csv("../input/train.csv")



test_set = pd.read_csv("../input/test.csv")



gender_set = pd.read_csv("../input/gender_submission.csv")



train_set.head()
test_set.head()
gender_set.head()
train_label = train_set['Survived'].copy()

train_label.head()
train_set = train_set.drop(['PassengerId', 'Survived', 'Cabin'], axis=1)

train_set.head()
train_set.describe()
train_set.info()
test_set = test_set.drop(['PassengerId', 'Cabin'], axis=1)

test_label = gender_set.drop(['PassengerId'], axis=1)
num_features = ['Age', 'Fare']

cat_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

str_features = ['Name', 'Ticket']
train_label.describe()
train_label.value_counts()
from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]

      

      

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer



num_pipeline = Pipeline([

        ("select_numeric", DataFrameSelector(num_features)),

        ("imputer", SimpleImputer(strategy="median")),

    ])



num_pipeline.fit_transform(train_set)

class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                        index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)
from sklearn.preprocessing import OneHotEncoder



cat_pipeline = Pipeline([

    ("cat_features_selection", DataFrameSelector(cat_features)),

    ("fill_with_most_frequent", MostFrequentImputer()),

    ("one_hot_encoder", OneHotEncoder(sparse=False))

])
both_set = train_set.append(test_set, ignore_index=True)

both_set.info()
from sklearn.pipeline import FeatureUnion



preprocess_pipeline = FeatureUnion(transformer_list=[

        ("num_pipeline", num_pipeline),

        #("most_frequent_pipeline", most_frequent_pipeline),

        ("cat_pipeline", cat_pipeline),

    ], n_jobs=1)



preprocess_pipeline.fit(both_set)

X_train = preprocess_pipeline.transform(train_set)

X_test = preprocess_pipeline.transform(test_set)

y_test = test_label.values

y_train = train_label.values

X_train
from sklearn.svm import SVC





svm_clf = SVC()

svm_clf.fit(X_train, y_train)

survived = svm_clf.predict([X_train[42]])

print('this guy survided? {}, we predicted {}'.format(y_train[42], survived))
from sklearn.model_selection import cross_val_score



svm_clf_cross_val = cross_val_score(estimator=svm_clf, X=X_train, y=y_train, cv=5)



svm_clf_cross_val.mean()
from sklearn.naive_bayes import GaussianNB



GNB_clf = GaussianNB()



GNB_clf_cross_val = cross_val_score(estimator=GNB_clf, X=X_train, y=y_train, cv=5)



GNB_clf_cross_val.mean()
from sklearn.neighbors import KNeighborsClassifier



KNN_clf = KNeighborsClassifier()



KNN_clf_cross_val = cross_val_score(estimator=KNN_clf, X=X_train, y=y_train, cv=5)

KNN_clf_cross_val.mean()
from sklearn.ensemble import RandomForestClassifier



RF_clf = RandomForestClassifier()



RF_clf_cross_val = cross_val_score(estimator=RF_clf, X=X_train, y=y_train, cv=5)

print('score: {}'.format(RF_clf_cross_val.mean()))

# let's focus on RandomForest classifier
from sklearn.metrics import accuracy_score

import numpy as np

RF_clf.fit(X_train, y_train)



y_predict = RF_clf.predict(X_test)



print('score on test_set: {}'.format(accuracy_score(y_test, y_predict)))
from sklearn.model_selection import RandomizedSearchCV

import scipy as sp



param_distributions={'n_estimators': sp.stats.randint(2, 10),

                     'max_features': sp.stats.randint(3, 10),

                     #'bootstrap': [True, False],

}







search = RandomizedSearchCV(RF_clf, param_distributions, cv = 10, n_iter=100, scoring = 'accuracy', random_state=42)



search.fit(X_train, y_train)





print(search.best_score_)

print(search.best_params_)



y_predict = search.best_estimator_.predict(X_test)

sum_good_predict = sum(y_predict == y_test.reshape([-1]))

score = (sum_good_predict/len(y_predict))

print('score on test_set: {}'.format(score))
