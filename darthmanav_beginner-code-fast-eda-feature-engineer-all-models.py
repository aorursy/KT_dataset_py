!pip install sweetviz
import numpy as np
import pandas as pd
import pandas_profiling as pp
import math
import random
import seaborn as sns
import sweetviz as sv
from IPython.display import Image
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import missingno as msno
%matplotlib inline

# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, ShuffleSplit
from sklearn.model_selection import cross_val_predict as cvp
from sklearn import metrics, pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score


from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from time import time
# models
from sklearn.linear_model import LogisticRegression, LogisticRegression, Perceptron, RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
submit=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.head()
train.isnull().sum()
msno.matrix(train)
msno.matrix(test)
dataset = [train,test]

for data in dataset:
    # coplete missing age with median
    data['Age'].fillna(data['Age'].median(),inplace = True)
    
    # complete Embarked with mode
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)
    
    # complete missing Fare with median
    data['Fare'].fillna(data['Fare'].median(),inplace = True)
print("Train info:")
print(train.isnull().sum())
print()
print()
print("Test info:")
print(test.isnull().sum())
train.drop(['Cabin','PassengerId'], axis=1, inplace = True)
test.drop(['Cabin','PassengerId'],axis=1,inplace=True)
train.head()
print("Train info:")
print(train.isnull().sum())
print()
print()
print("Test info:")
print(test.isnull().sum())
report=pp.ProfileReport(train)
report
#analyzing the dataset
analysis=sv.analyze(train)
analysis.show_html('train_analysis.html')
def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
numVar = ["Fare", "Age"]
for n in numVar:
    plot_hist(n)
sns.pairplot(train)
plt.show()
sns.heatmap(train.corr(), annot = True, fmt = ".2f")
plt.show()
g = sns.factorplot(x = "SibSp", y = "Survived", data = train, kind = "bar", size = 6)
g.set_ylabels("Survived Probability")
plt.show()
g = sns.factorplot(x = "Parch", y = "Survived", kind = "bar", data = train, size = 6)
g.set_ylabels("Survived Probability")
plt.show()
g = sns.FacetGrid(train, col = "Survived",size=6)
g.map(sns.distplot, "Age", bins = 25)
plt.show()
train.drop(['Embarked'], axis=1, inplace = True)
test.drop(['Embarked'],axis=1,inplace=True)
genders = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(genders)
test['Sex'] = test['Sex'].map(genders)
train['Sex'].value_counts()
for data in dataset:
    data['Age'] = data['Age'].astype(int)
    data.loc[ data['Age'] <= 15, 'Age'] = 0
    data.loc[(data['Age'] > 15) & (data['Age'] <= 20), 'Age'] = 1
    data.loc[(data['Age'] > 20) & (data['Age'] <= 26), 'Age'] = 2
    data.loc[(data['Age'] > 26) & (data['Age'] <= 28), 'Age'] = 3
    data.loc[(data['Age'] > 28) & (data['Age'] <= 35), 'Age'] = 4
    data.loc[(data['Age'] > 35) & (data['Age'] <= 45), 'Age'] = 5
    data.loc[ data['Age'] > 45, 'Age'] = 6
train['Age'].value_counts()
train.head()
for data in dataset:
    data['Family'] = data['SibSp'] + data['Parch'] + 1
train.head()    
for data in dataset:
    drop_column = ['Fare','Name','Ticket','SibSp','Parch']
    data.drop(drop_column, axis=1, inplace = True)
train.head()
test.head()
X_train,X_val,y_train,y_val=train_test_split(train.iloc[:,1:],train['Survived'],test_size=0.2,random_state=2)
def acc_summary(pipeline, X_train, y_train, X_val, y_val):
    t0 = time()
    sentiment_fit = pipeline.fit(X_train, y_train)
    y_pred = sentiment_fit.predict(X_val)
    train_test_time = time() - t0
    accuracy = accuracy_score(y_val, y_pred)*100
    print("accuracy : {0:.2f}%".format(accuracy))
    print("train and test time: {0:.2f}s".format(train_test_time))
    print("-"*80)
    return accuracy, train_test_time

names = [ 
        'Logistic Regression',
        'Ridge Classifier',
        'SGD Classifier',
        'SVC',
        'Gradient Boosting Classifier', 
        'Extra Trees Classifier', 
        "Bagging Classifier",
        "AdaBoost Classifier", 
        "K Nearest Neighbour Classifier",
         "Decison Tree Classifier",
         "Random Forest Classifier",
         'GaussianNB',
        "Gaussian Process Classifier",
        "MLP Classifier",
        "XGB Classifier",
        "LGBM Classifier"
         ]
classifiers = [
    LogisticRegression(),
    RidgeClassifier(),
    SGDClassifier(),
    SVC(),
    GradientBoostingClassifier(),
    ExtraTreesClassifier(), 
    BaggingClassifier(),
    AdaBoostClassifier(),
    KNeighborsClassifier(n_neighbors=3),
    DecisionTreeClassifier(max_depth=3),
    RandomForestClassifier(n_estimators=100),
    GaussianNB(),
    GaussianProcessClassifier(),
    MLPClassifier(),
    XGBClassifier(booster= 'dart', max_depth=2,n_estimators=500),
    LGBMClassifier()
        ]

zipped_clf = zip(names,classifiers)
def classifier_comparator(X_train,y_train,X_val,y_val,classifier=zipped_clf): 
    result = []
    for n,c in classifier:
        checker_pipeline = Pipeline([
            ('classifier', c)
        ])
        print("Validation result for {}".format(n))
        print(c)
        clf_acc,tt_time = acc_summary(checker_pipeline,X_train, y_train, X_val, y_val)
        result.append((n,clf_acc,tt_time))
    return result
classifier_comparator(X_train,y_train,X_val,y_val)
model=RandomForestClassifier(n_estimators=200)
model.fit(train.iloc[:,1:],train['Survived'])
y_pred=model.predict(test)
submit['Survived']=y_pred
#submit.to_csv('submission_file.csv',index=False)