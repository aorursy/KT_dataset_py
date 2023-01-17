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
# Load the training data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Split the labels into one array and data into another
train_labels = train['Survived']
train_data = train.drop(['Survived'], axis=1)
# Split the dataset into train and test
# Let's make training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import minmax_scale, MinMaxScaler

def convert_sex(sex):
    return 0 if sex == 'male' else 1

def get_title(name):
    return name.split(",")[1].split(".")[0][1:]

def get_title_code(name):
    known_titles = ['Mr', 'Mrs', 'Miss', 'Master','Don','Rev','Dr','Mme','Ms','Major','Lady','Sir','Mlle','Col','Capt','the Countess','Jonkheer']
    title = get_title(name)
    if title in known_titles:
        return known_titles.index(title)
    return -1

def extract_features(df, scale):
    feat = pd.DataFrame()
    feat["sex"] = df['Sex'].apply(convert_sex)
    median_fare = df['Fare'].median()
    feat["fare"] = df["Fare"].fillna(median_fare)
    median_age = df['Age'].median()
    feat["age_set"] = df['Age'].notnull().apply(int)
    feat["age"] = df['Age'].fillna(median_age)
    feat["class"] = df["Pclass"]
    
    # Scale all features together
    feat["title_code"] = df['Name'].apply(get_title_code)
    if scale:
        feat[feat.columns] = MinMaxScaler().fit_transform(feat[feat.columns])
    return feat
# Make the feature extractor pipeline-able
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, scale=True):
        self.scale = scale
    
    def fit( self, X, y = None):
        return self
    
    def transform( self, X):
        return extract_features(X, self.scale)
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import GridSearchCV

# Find the best model
lgb = Pipeline(
    [('extract', FeatureExtractor(False)),
     ('lgbmc', LGBMClassifier(random_state=42, boosting_type='goss', max_depth=4))])

# parameters = {
#     'lgbmc__boosting_type':['gbdt','dart','goss'],
#     'lgbmc__max_depth':[-1,2,3,4,5,20],
#     'lgbmc__n_estimators':[100,1000]
# }
# lgb = GridSearchCV(lgb, parameters, cv=10).fit(X_train, y_train)
# print(lgb)
# print(lgb.best_estimator_)
# lgb=lgb.best_estimator_

# cross-validate with default params
y_train_pred = cross_val_predict(lgb, X_train, y_train, cv=10)
print('The training CV accuracy of prediction is:', accuracy_score(y_train, y_train_pred))

# check against validation set
lgb.fit(X_train, y_train)
y_pred = lgb.predict(X_test)
print('The accuracy of prediction against validation set is:', accuracy_score(y_test, y_pred))

# Prepare submission
lgb.fit(train_data, train_labels)
test_pred = lgb.predict(test)
results = test[['PassengerId']]
results['Survived'] = test_pred[results.index]
print(results)
results.to_csv("submission.csv",index=False)
