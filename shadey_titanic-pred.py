import sys

sys.version
import pandas as pd

import numpy as np 

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()
train.info()
stat = train.describe()

stat
train['Survived'].value_counts()
train['Pclass'].value_counts()
train['Sex'].value_counts()
train['Embarked'].value_counts()
train["RelativesOnboard"] = train["SibSp"] + train["Parch"]

train[["RelativesOnboard", "Survived"]].groupby(['RelativesOnboard']).mean()
train["AgeBucket"] = train["Age"] // 15 * 15

train[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()
corr = train.corr()

import seaborn as sns

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]



    

# Let's build the pipeline for the numerical attributes:



from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer



num_pipeline = Pipeline([

        ('select_numeric', DataFrameSelector(['Age', 'SibSp', 'Parch', 'Fare'])),

        ('imputer', SimpleImputer(strategy='median')),

        ])



num_pipeline.fit_transform(train)

    
class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                         index = X.columns)

        return self

    def transform(self, X, y = None):

        return X.fillna(self.most_frequent_)

    

from sklearn.preprocessing import OneHotEncoder



cat_pipeline = Pipeline([

        ('Select_cat', DataFrameSelector(['Pclass', 'Sex', 'Embarked'])),

        ('imputer', MostFrequentImputer()),

        ('cat_encoder',OneHotEncoder(sparse=False)),

        ])    

    

cat_pipeline.fit_transform(train)
from sklearn.pipeline import FeatureUnion

preprocess_pipeline = FeatureUnion(transformer_list= [

            ('num_pipeline', num_pipeline),

            ('cat_pipeline', cat_pipeline),

        ])
X_train = preprocess_pipeline.fit_transform(train)

y_train = train['Survived']
X_train.shape
from sklearn.model_selection import GridSearchCV
#SGD loss = 'log' implements Logistic regression

from sklearn.linear_model import SGDClassifier

sgd_clf_log = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42, loss = 'log')

sgd_clf_log.fit(X_train, y_train)



#SGD loss = 'hinge' implements Linear SVM.

from sklearn.linear_model import SGDClassifier

sgd_clf_svm = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42, loss = 'hinge')

sgd_clf_svm.fit(X_train, y_train)





#SVC

from sklearn.svm import SVC

svc_clf = SVC(gamma = 'auto')

svc_clf.fit(X_train, y_train)





#Random Forest

param_grid_forest = [{'n_estimators': [50, 100, 200]}]

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)

grid_search_forest = GridSearchCV(forest_clf, param_grid_forest, cv=5, verbose=0)

grid_search_forest.fit(X_train, y_train)





#KNN

param_grid_knn = [{'weights': ["uniform", "distance"], 'n_neighbors': [ 2, 3, 4]}]

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()

grid_search_knn = GridSearchCV(knn_clf, param_grid_knn, cv=5, verbose=0)

grid_search_knn.fit(X_train, y_train)



#XGD 

import xgboost as xgb

xgb_clf = xgb.XGBClassifier(

    #learning_rate = 0.02,

 n_estimators= 2000,

 max_depth= 4,

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1).fit(X_train, y_train)
grid_search_forest.best_params_
grid_search_knn.best_params_
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict
# Logistic Regression accuracy

svm_scores_log = cross_val_score(sgd_clf_log, X_train, y_train, cv=10)

svm_scores_log.mean()
# SVM accuracy

svm_scores_svm = cross_val_score(sgd_clf_svm, X_train, y_train, cv=10)

svm_scores_svm.mean()
# SVC Accuracy

svc_scores = cross_val_score(svc_clf, X_train, y_train, cv=10)

svc_scores.mean()
# K-nearest Neighbors (KNN) accuracy

knn_scores = cross_val_score(grid_search_knn, X_train, y_train, cv=10)

knn_scores.mean()
# Random Forest accuracy

forest_scores = cross_val_score(grid_search_forest, X_train, y_train, cv=10)

forest_scores.mean()
y_train_forest_pred = cross_val_predict(grid_search_forest, X_train, y_train, cv=3)

confusion_matrix(y_train, y_train_forest_pred)
# XGBoost (XGB) accuracy

xgb_scores = cross_val_score(xgb_clf, X_train, y_train, cv=10)

xgb_scores.mean()
y_train_xgb_pred = cross_val_predict(xgb_clf, X_train, y_train, cv=3)

confusion_matrix(y_train, y_train_xgb_pred)
import matplotlib.pyplot as plt 

plt.figure(figsize=(10, 6))

plt.plot([1]*10, svc_scores, '.')

plt.plot([2]*10, forest_scores, '.')

plt.plot([3]*10, xgb_scores, '.')

plt.plot([4]*10, knn_scores, '.')

plt.plot([5]*10, svm_scores_svm, '.')

plt.plot([6]*10, svm_scores_log, '.')

plt.boxplot([svc_scores, forest_scores, xgb_scores, knn_scores, svm_scores_svm, svm_scores_log], labels=('SVC', 'Random Forest', 'XGBoost', 'KNN', 'SVM', 'Logistic Regression'))

plt.ylabel('Accuracy', fontsize=14)

plt.show()
X_test = preprocess_pipeline.transform(test)

xgb_pred = xgb_clf.predict(X_test)

y_pred_forest = grid_search_forest.predict(X_test)
test['Survived'] = xgb_pred.astype(int)

forest_results = test[['PassengerId','Survived']].to_csv("titanic_xgb_clf.csv",index=False)
test['Survived'] = y_pred_forest.astype(int)

forest_results = test[['PassengerId','Survived']].to_csv("titanic_forest_clf.csv",index=False)