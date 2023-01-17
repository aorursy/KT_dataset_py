# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data = pd.read_csv("../input/unimelb_training.csv")
test = pd.read_csv("../input/unimelb_test.csv")

# Any results you write to the current directory are saved as output.
data.shape
data.head()
data.info()
data.columns
data.describe()
total = data.isnull().sum().sort_values(ascending = False)
total
percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending = False)
missing_train_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Tercent'])
missing_train_data.head(50)
data.dropna(axis = 1, how = 'all')
data.dropna(axis = 1, how = 'all', thresh = 8000, inplace = True)
train_columns = data.columns
test.dropna(axis = 1, how = 'all', thresh = 8000)
test.head()
test.shape
data.shape
data.info()
numerical_columns = [col for col in data.columns if data[col].dtype != 'object']
numerical_columns
categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
categorical_columns
input_vars = numerical_columns
input_vars
input_vars.remove('Grant.Status')
input_vars
input_vars.remove('Grant.Application.ID')
output_var = 'Grant.Status'
from sklearn.linear_model import LogisticRegression
logit_regr1 = LogisticRegression()
input_data = data
logit_regr1.fit(X=input_data[input_vars], y=input_data[output_var])
input_data.fillna((input_data.mean()), inplace=True)
input_data.info()
logit_regr1.fit(X=input_data[input_vars], y=input_data[output_var])
from sklearn.metrics import accuracy_score
preds = logit_regr1.predict(X=input_data[input_vars])
print(accuracy_score(y_pred=preds, y_true=input_data[output_var]))
from sklearn.ensemble import RandomForestClassifier
random_forest1 = RandomForestClassifier(n_estimators=5, max_depth=5)
random_forest1.fit(X=input_data[input_vars], y=input_data[output_var])
print(accuracy_score(y_pred=random_forest1.predict(X=input_data[input_vars]), y_true=input_data[output_var]))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred=random_forest1.predict(X=input_data[input_vars]), y_true=input_data[output_var])
from sklearn.metrics import roc_curve, auc
random_forest1.predict_proba(input_data[input_vars])[:, 1]
rf_1s_preds = random_forest1.predict_proba(input_data[input_vars])[ :, 1]
fpr, tpr, threshold = roc_curve(y_true=input_data[output_var], y_score=rf_1s_preds)
roc_auc = auc(fpr, tpr)
roc_auc
from sklearn.model_selection import train_test_split




