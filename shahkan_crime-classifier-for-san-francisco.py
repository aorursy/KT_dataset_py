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
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score as score

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


train_data =pd.read_csv("../input/san-francisco-crime-classification/train.csv", parse_dates =['Dates'])

test_data = pd.read_csv("../input/sf-crime/test.csv", parse_dates =['Dates'])

print("The size of the train data is:", train_data.shape)

print("The size of the test data is:", test_data.shape)
train_data.head()
test_data.head()
train_data.dtypes.value_counts()
test_data.dtypes.value_counts()

train_data.isnull().sum()
test_data.isnull().sum()
train_data.columns
test_data.columns
train_data.Category.value_counts()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_data['Category'] = le.fit_transform(train_data.Category)

train_data.Category.head()
train_data.PdDistrict.value_counts()
feature_cols =['DayOfWeek', 'PdDistrict']

train_data = pd.get_dummies(train_data, columns=feature_cols)

test_data = pd.get_dummies(test_data, columns=feature_cols)



train_data
test_data
for x in [train_data, test_data]:

    x['years'] = x['Dates'].dt.year

    x['months'] = x['Dates'].dt.month

    x['days'] = x['Dates'].dt.day

    x['hours'] = x['Dates'].dt.hour

    x['minutes'] = x['Dates'].dt.minute

    x['seconds'] = x['Dates'].dt.second
train_data.head()
test_data.head()
train_data = train_data.drop(['Dates', 'Address','Resolution'], axis = 1)

train_data = train_data.drop(['Descript'], axis = 1)

train_data.head()
test_data = test_data.drop(['Dates', 'Address'], axis = 1)

test_data.head()
feature_cols = [x for x in train_data if x!='Category']

X = train_data[feature_cols]

y = train_data['Category']

X_train, x_test,y_train, y_test = train_test_split(X, y)
DTC = DecisionTreeClassifier(criterion = 'gini', max_features = 10, max_depth = 5)

DTC = DTC.fit(X_train, y_train)

y_pred_DTC = DTC.predict(x_test)

y_pred_test_DTC = DTC.predict(X_train)



print("score is {:.3f}".format (score(y_test, y_pred_DTC, average = 'micro')*100))

print("Accuracy for the test data is {:.3f} ".format (accuracy_score(y_test, y_pred_DTC)*100))

print("Accuracy for the train data is {:.3f} ".format (accuracy_score(y_train, y_pred_test_DTC)*100))