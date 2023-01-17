#Importing dependencies



# Standard Python Imports

from timeit import default_timer as timer

import time, datetime

import os



# Data Manipulation

import numpy as np

import pandas as pd



# Data Visualization 

import matplotlib.pyplot as plt

import missingno as msno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



# For Data Preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer

from sklearn.preprocessing import Imputer

from sklearn.pipeline import Pipeline

try:

    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+

except ImportError:

    from sklearn.preprocessing import Imputer as SimpleImputer



# Machine Learning

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import FeatureUnion

# Algorithms

from sklearn.impute import SimpleImputer

from sklearn.svm import LinearSVC

from sklearn import svm

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost.sklearn import XGBClassifier

# Validation & Scoring

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

# Ignoring the warnings that we will see in this notebook

import warnings

warnings.filterwarnings('ignore')
# Importing Data

def fetch_data():

    data = pd.read_csv("./../input/mushrooms.csv")

    return data

data = fetch_data()
msno.matrix(data)
data.describe()
data.head()
data.tail()
data.get_dtype_counts()
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(data, data["class"]):

    train = data.loc[train_index]

    test = data.loc[test_index]
sns.countplot(y="class", data=data)
sns.countplot(y="class", data=train)
sns.countplot(y="class", data=test)
Y_train = train["class"]

X_train = train.iloc[:,1:]



Y_test = test["class"]

X_test = test.iloc[:,1:]
X_train.head()
class DropColumns(BaseEstimator, TransformerMixin):

    def __init__(self, column_names=[]):

        self.column_names = column_names

    def transform(self, df, y=None):

        return df.drop(self.column_names, axis=1)

    def fit(self, df, y=None):

        return self
class ColumnExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, column_names=[]):

        self.column_names = column_names

    def transform(self, df, y=None):

        return df.loc[:, self.column_names]

    def fit(self, df, y=None):

        return self
class FeatureNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, column_names=[]):

        self.column_names = column_names

        self.min_max_scalar = MinMaxScaler()

    def fit(self, X, y=None):

        self.min_max_scalar.fit(X[self.column_names])

        return self

    def transform(self, X, y=None):

        X[self.column_names] = self.min_max_scalar.transform(X[self.column_names])

        return X
class MissingStalkRoots(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    def fit(self, X, y=None):

        X["stalk-root"] = X["stalk-root"].replace(['?'], 'm')

        return self

    def transform(self, X, y=None):

        return X
class ReplacingVeilColor(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    def fit(self, X, y=None):

        X["veil-color"] = X["veil-color"].replace(['n', 'o'], 'nw')

        return self

    def transform(self, X, y=None):

        return X
class MyLabelBinarizer(TransformerMixin):

    def __init__(self):

        self.encoder = LabelBinarizer()

    def fit(self, x, y=0):

        self.encoder.fit(x)

        return self

    def transform(self, x, y=0):

        return self.encoder.transform(x)
sns.countplot(y="cap-shape", data=train)
sns.countplot(y="cap-surface", data=train)
sns.countplot(y="cap-color", data=train)
sns.countplot(y="bruises", data=train)
sns.countplot(y="odor", data=train)
sns.countplot(y="gill-spacing", data=train)
sns.countplot(y="gill-size", data=train)
sns.countplot(y="stalk-shape", data=train)
sns.countplot(y="stalk-root", data=train)
sns.countplot(y="stalk-surface-above-ring", data=train)
sns.countplot(y="stalk-surface-below-ring", data=train)
sns.countplot(y="stalk-color-above-ring", data=train)
sns.countplot(y="stalk-color-below-ring", data=train)
sns.countplot(y="veil-color", data=train)
sns.countplot(y="ring-number", data=train)
counter = train[train["ring-number"] == "n"]

print("There are {} ring-numbers with value \'n\'".format(len(counter)))
sns.countplot(y="ring-type", data=train)
data_preprocessing = Pipeline([

        ("drop_veil_tape", DropColumns(["veil-type"])),

        ("replacing_stalk_roots", MissingStalkRoots()),

        ("repalcing_veil_color", ReplacingVeilColor()),

        #("one_hot_encoding", OneHotEncoder(sparse=False))

    ])



labels_preprocessing = Pipeline([

        ("one_hot_encoding", MyLabelBinarizer())

    ])
X_train = data_preprocessing.fit_transform(X_train)

X_test = data_preprocessing.fit_transform(X_test)

Y_train = labels_preprocessing.fit_transform(Y_train)

Y_test = labels_preprocessing.fit_transform(Y_test)
combinedsets = pd.concat([X_train, X_test])

enc = OneHotEncoder(sparse=False)

enc.fit(combinedsets)

X_train = enc.transform(X_train)

X_test = enc.transform(X_test)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
# Function that runs the requested algorithm and returns the accuracy metrics

def fit_ml_algo(algo, x_train, y_train, cv):

    # One Pass

    model = algo.fit(x_train, y_train)

    acc = round(model.score(x_train, y_train) * 100, 2)

    

    # Cross Validation 

    train_pred = model_selection.cross_val_predict(algo, 

                                                  x_train, 

                                                  y_train, 

                                                  cv=cv, 

                                                  n_jobs = -1)

    # Cross-validation accuracy metric

    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)

    

    #print("Model used :", algo.best_estimator_)

    return train_pred, acc, acc_cv
class MachineLearningClassification(TransformerMixin):

    def __init__(self, x_train, y_train):

        self.x_train = x_train

        self.y_train = y_train

    def fit(self, x_train, y_train):

        """knn_params = {'n_neighbors':list(range(1,100)), 'weights': ['distance', 'uniform']}

        knn_grid_search_cv = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)

        knn_grid_search_cv.fit(self.x_train, self.y_train)

        knn = knn_grid_search_cv.best_estimator_

        

        Cs = [0.001, 0.01, 0.1, 1, 10]

        gammas = [0.001, 0.01, 0.1, 1]

        kernels = ['rbf', 'linear']

        param_grid = {'C': Cs, 'gamma' : gammas, 'kernel': kernels}

        svm_grid_search = GridSearchCV(svm.SVC(), param_grid, cv=10)

        svm_grid_search.fit(self.x_train, self.y_train)

        svmc = svm_grid_search.best_estimator_

        

        rf_params = {'n_estimators': list(range(1,100))}

        rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_params, cv=10)

        rf_grid_search.fit(self.x_train, self.y_train)

        rfc = rf_grid_search.best_estimator_"""

        

        sgdc_params = {"loss": ["hinge", "log"], "penalty": ["l1", "l2"], "max_iter": [1,2,3,4,5]}

        sgdc_grid_search = GridSearchCV(SGDClassifier(), sgdc_params, cv=5)

        sgdc_grid_search.fit(self.x_train, self.y_train)

        sgdc = sgdc_grid_search.best_estimator_

                       

        gbc_params = {"loss": ["deviance", "exponential"],"learning_rate": [1,0.6 ,0.5,0.4,0.3, 0.25, 0.1, 0.05, 0.01],"n_estimators": [10,50,100]}

        gbc_grid_search = GridSearchCV(GradientBoostingClassifier(), gbc_params, cv=5)

        gbc_grid_search.fit(self.x_train, self.y_train)

        gbc = gbc_grid_search.best_estimator_

                       

        """ lsvc_params = {"penalty": ["l2"],"loss": ["hinge", "squared_hinge"],"dual": [True],"C": [0.001,0.01,0.1,1,10]}     

        lsvc_grid_search = GridSearchCV(LinearSVC(), lsvc_params, cv=5)

        lsvc_grid_search.fit(self.x_train, self.y_train)

        lsvc = lsvc_grid_search.best_estimator_

                       

        xgb_params = {"early_stopping_rounds": [1,2,5],"n_estimators": [5,10,15],"learning_rate": [0.001,0.03,0.05],"n_jobs": [0,1,2,5]}

        xgb_grid_search = GridSearchCV(XGBClassifier(), xgb_params, cv=5)

        xgb_grid_search.fit(self.x_train, self.y_train)

        xgb = xgb_grid_search.best_estimator_

        """

                       

        classifiers_array = [LogisticRegression(),

                             #knn,

                             #svmc,

                             DecisionTreeClassifier(),

                             #rfc,

                             sgdc,

                             gbc,

                             #lsvc,

                             #xgb

                            ]          

        

        best_cls = None

        best_acc = None

        best_acc_cv = None

        accs = []

        accs_cv = []

        

        for clf in classifiers_array:

            train_pred, clf_acc, clf_acc_cv = fit_ml_algo(clf,self.x_train,self.y_train,5)

            accs.append(clf_acc)

            accs_cv.append(clf_acc_cv)

            

        best_acc = max(accs)

        best_acc_cv = max(accs_cv)

        best_cls = classifiers_array[accs_cv.index(best_acc_cv)]

        return best_acc, best_acc_cv, best_cls

    def transform(self, x_train, y_train):

        return best_acc, best_acc_cv, best_cls
def classify(x_train, y_train):

    """knn_params = {'n_neighbors':list(range(1,100)), 'weights': ['distance', 'uniform']}

    knn_grid_search_cv = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)

    knn_grid_search_cv.fit(x_train, y_train)

    knn = knn_grid_search_cv.best_estimator_

        

    Cs = [0.001, 0.01, 0.1, 1, 10]

    gammas = [0.001, 0.01, 0.1, 1]

    kernels = ['rbf', 'linear']

    param_grid = {'C': Cs, 'gamma' : gammas, 'kernel': kernels}

    svm_grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5)

    svm_grid_search.fit(x_train, y_train)

    svmc = svm_grid_search.best_estimator_

        

    rf_params = {'n_estimators': list(range(1,100))}

    rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_params, cv=5)

    rf_grid_search.fit(x_train, y_train)

    rfc = rf_grid_search.best_estimator_"""

        

    sgdc_params = {"loss": ["hinge", "log"], "penalty": ["l1", "l2"], "max_iter": [1,2,3,4,5]}

    sgdc_grid_search = GridSearchCV(SGDClassifier(), sgdc_params, cv=5)

    sgdc_grid_search.fit(x_train, y_train)

    sgdc = sgdc_grid_search.best_estimator_

                       

    gbc_params = {"loss": ["deviance", "exponential"],"learning_rate": [1,0.6 ,0.5,0.4,0.3, 0.25, 0.1, 0.05, 0.01],"n_estimators": [10,50,100]}

    gbc_grid_search = GridSearchCV(GradientBoostingClassifier(), gbc_params, cv=5)

    gbc_grid_search.fit(x_train, y_train)

    gbc = gbc_grid_search.best_estimator_

                       

    """ lsvc_params = {"penalty": ["l2"],"loss": ["hinge", "squared_hinge"],"dual": [True],"C": [0.001,0.01,0.1,1,10]}     

    lsvc_grid_search = GridSearchCV(LinearSVC(), lsvc_params, cv=5)

    lsvc_grid_search.fit(x_train, y_train)

    lsvc = lsvc_grid_search.best_estimator_

                       

    xgb_params = {"early_stopping_rounds": [1,2,5],"n_estimators": [5,10,15],"learning_rate": [0.001,0.03,0.05],"n_jobs": [0,1,2,5]}

    xgb_grid_search = GridSearchCV(XGBClassifier(), xgb_params, cv=5)

    xgb_grid_search.fit(x_train, y_train)

    xgb = xgb_grid_search.best_estimator_

    """

                       

    classifiers_array = [LogisticRegression(),

                        #knn,

                        #svmc,

                        DecisionTreeClassifier(),

                        #rfc,

                        sgdc,

                        gbc,

                        #lsvc,

                        #xgb

                        ]          

        

    best_cls = None

    best_acc = None

    best_acc_cv = None

    accs = []

    accs_cv = []

        

    for clf in classifiers_array:

        train_pred, clf_acc, clf_acc_cv = fit_ml_algo(clf,x_train,y_train,5)

        accs.append(clf_acc)

        accs_cv.append(clf_acc_cv)

            

    best_acc = max(accs)

    best_acc_cv = max(accs_cv)

    best_cls = classifiers_array[accs_cv.index(best_acc_cv)]

    return best_acc, best_acc_cv, best_cls
classification = Pipeline([

        ("classification_best", MachineLearningClassification(X_train, Y_train))

    ])
best_classifier,best_acc_cv, best_acc = classify(X_train, Y_train)

print("Best Classifier : ", best_classifier)

print("Best Acc, CV : ", best_acc_cv)

print("Best Acc : ", best_acc)
model = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,

            max_features=None, max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, presort=False, random_state=None,

            splitter='best')

model.fit(X_train, Y_train)

pred = model.predict(X_test)

print("Test accuracy :", round(metrics.accuracy_score(Y_test, pred) * 100, 2))
from sklearn.metrics import precision_score, recall_score, f1_score

print("Precision : ", precision_score(Y_test, pred))

print("Recall : ", recall_score(Y_test, pred))

print("F1 : ", f1_score(Y_test, pred))