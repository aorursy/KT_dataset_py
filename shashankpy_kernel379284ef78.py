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

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler



from imblearn.over_sampling import SMOTE
from sklearn import tree
import graphviz
from sklearn import metrics

import matplotlib.pyplot as plt
from IPython.display import Image
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
cc_fraud_data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
cc_fraud_data.shape, cc_fraud_data.columns, cc_fraud_data.dtypes
cc_fraud_data.describe()
cc_fraud_data.info()
cc_fraud_data.columns
cc_fraud_data.Class.value_counts()
cc_fraud_data.isnull().sum()
features = list(cc_fraud_data.columns)[:-1]
target = list(cc_fraud_data.columns)[-1]
features, target
X = cc_fraud_data[features]
y = cc_fraud_data[target]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(pd.value_counts(y_train)/y_train.astype('object').shape[0] * 100)

print(pd.value_counts(y_test) /y_test.astype('object').shape[0] * 100)
log_regr = LogisticRegression()
model_train = log_regr.fit(X_train, y_train)
model_result = log_regr.predict(X_test)
model_accuracy = accuracy_score(model_result, y_test)
model_recall = recall_score(model_result, y_test)
model_precision = precision_score(model_result, y_test)
model_f1 = f1_score(model_result, y_test)
nl = "\n"
print(f"        The logistic regression model has Accuracy  : {model_accuracy : 10}{nl}\
        The logistic regression model has Recall    :  {model_recall:10}{nl}\
        The logistic regression model has Precision : {model_precision : 10}{nl}\
        The logistic regression model has F1        : {model_f1 : 10}")

print(confusion_matrix(model_result, y_test))
log_mod2 = LogisticRegression(class_weight='balanced', random_state=123)
log_mod2.fit(X_train, y_train)
model_result2 = log_mod2.predict(X_test)
model_accuracy = accuracy_score(model_result2, y_test)
model_recall = recall_score(model_result2, y_test)
model_precision = precision_score(model_result2, y_test)
model_f1 = f1_score(model_result2, y_test)
nl = "\n"
print(f"        The logistic regression model has Accuracy  : {model_accuracy : 10}{nl}\
        The logistic regression model has Recall    :  {model_recall:10}{nl}\
        The logistic regression model has Precision : {model_precision : 10}{nl}\
        The logistic regression model has F1        : {model_f1 : 10}")
print(confusion_matrix(model_result2, y_test))
smote = SMOTE(random_state=123)
X_train_sm, y_train_sm = smote.fit_sample(X_train, y_train)
log_regr.fit(X_train_sm, y_train_sm)
model_result3 = log_regr.predict(X_test)
model_accuracy = accuracy_score(model_result3, y_test)
model_recall = recall_score(model_result3, y_test)
model_precision = precision_score(model_result3, y_test)
model_f1 = f1_score(model_result3, y_test)
nl = "\n"
print(f"        The logistic regression model has Accuracy  : {model_accuracy : 10}{nl}\
        The logistic regression model has Recall    :  {model_recall:10}{nl}\
        The logistic regression model has Precision : {model_precision : 10}{nl}\
        The logistic regression model has F1        : {model_f1 : 10}")
dtclf = tree.DecisionTreeClassifier(max_depth=3)
dtclf.fit(X_train, y_train)
importances = dtclf.feature_importances_
importances
model_result4 = dtclf.predict(X_test)
model_accuracy = accuracy_score(model_result4, y_test)
model_recall = recall_score(model_result4, y_test)
model_precision = precision_score(model_result4, y_test)
model_f1 = f1_score(model_result4, y_test)
nl = "\n"
print(f"        The logistic regression model has Accuracy  : {model_accuracy : 10}{nl}\
        The logistic regression model has Recall    :  {model_recall:10}{nl}\
        The logistic regression model has Precision : {model_precision : 10}{nl}\
        The logistic regression model has F1        : {model_f1 : 10}")
dtclf2 = tree.DecisionTreeClassifier(criterion='gini', max_depth=5)
dtclf2 = dtclf2.fit(X_train, y_train)
model_result5 = dtclf2.predict(X_test)
model_accuracy = accuracy_score(model_result5, y_test)
model_recall = recall_score(model_result5, y_test)
model_precision = precision_score(model_result5, y_test)
model_f1 = f1_score(model_result5, y_test)
nl = "\n"
print(f"        The logistic regression model has Accuracy  : {model_accuracy : 10}{nl}\
        The logistic regression model has Recall    :  {model_recall:10}{nl}\
        The logistic regression model has Precision : {model_precision : 10}{nl}\
        The logistic regression model has F1        : {model_f1 : 10}")
# set of parameters to test
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 5],
              "max_depth": [None, 2],
              "min_samples_leaf": [1, 5]
             }
dtclf3 = tree.DecisionTreeClassifier()
dtclf_grid = GridSearchCV(dtclf3, param_grid, cv=3)
dtclf_grid.fit(X_train, y_train)
dtclf_grid.best_params_
model_result6 = dtclf2.predict(X_test)
model_accuracy = accuracy_score(model_result6, y_test)
model_recall = recall_score(model_result6, y_test)
model_precision = precision_score(model_result6, y_test)
model_f1 = f1_score(model_result6, y_test)
nl = "\n"
print(f"        The logistic regression model has Accuracy  : {model_accuracy : 10}{nl}\
        The logistic regression model has Recall    :  {model_recall:10}{nl}\
        The logistic regression model has Precision : {model_precision : 10}{nl}\
        The logistic regression model has F1        : {model_f1 : 10}")
