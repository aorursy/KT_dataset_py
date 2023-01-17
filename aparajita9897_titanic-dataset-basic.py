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
train = pd.read_csv("../input/train.csv")
train.head()
train.isnull().sum()
test = pd.read_csv("../input/test.csv")
test.isnull().sum()
len(train)
train["Age"].median()
test["Age"].median()
train["Age"].fillna(30,inplace = True)
test["Age"].fillna(27,inplace = True)
train["Embarked"].mode()
train["Embarked"].fillna("S",inplace = True)
train.isnull().sum()
train_clean = train.drop(columns = "Cabin")
train_clean = train_clean.drop(columns = "Ticket")
train_clean = train_clean.drop(columns = "Name")
test_clean = test.drop(columns = "Cabin")
test_clean = test_clean.drop(columns = "Name")
test_clean = test_clean.drop(columns = "Ticket")
test_clean["Fare"].fillna(test_clean["Fare"].median(), inplace = True)
test_clean.isnull().sum()
test_clean_new = test_clean.drop(columns = "PassengerId")
test_clean_new = test_clean_new.drop(columns = "Fare")
train_clean.isnull().sum()
train_clean.info()
train_clean.head()
train_clean["Embarked"].unique()
import scipy as sc
import sklearn
from sklearn import preprocessing
import seaborn as sb
label = preprocessing.LabelEncoder()
train_clean = train_clean.drop(columns = "PassengerId")
train_clean['Sex_Code'] = label.fit_transform(train_clean['Sex'])
train_clean['Embarked_code'] = label.fit_transform(train_clean['Embarked'])
train_clean.head(10)
sb.heatmap(train_clean.corr(), annot=True, fmt=".2f")
train_clean = train_clean.drop(columns = "Fare")
train_clean.head()
columns_x = ["Pclass","Age","SibSp","Parch","Sex","Embarked"]
data_x = pd.get_dummies(train_clean[columns_x])
data_x.head()
data_x_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Sex_female', 'Sex_male',
       'Embarked_C', 'Embarked_Q', 'Embarked_S']


clean_data_x = pd.get_dummies(test_clean_new[columns_x])
clean_data_x.head()
Target = ["Survived"]
from sklearn import model_selection
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data_x[data_x_columns],train_clean[Target])
print ("train_clean: {}".format(train_clean.shape))
print ("Xtrain: {}".format(X_train.shape))
print ("Xtest: {}".format(X_test.shape))
print ("Ytrain: {}".format(y_train.shape))
print ("Ytest: {}".format(y_test.shape))
from sklearn import linear_model
y_train_list = y_train.values
model = sklearn.linear_model.LogisticRegressionCV()
x = model.fit(X_train, y_train)
print("train: ", x.score(X_train, y_train))
print("test:  ", x.score(X_test, y_test))
y_predicted = model.predict(X_test)
import scikitplot as skplt
import matplotlib.pyplot as plt

#skplt.metrics.plot_roc_curve(y_test, y_predicted)
skplt.metrics.plot_confusion_matrix(y_test, y_predicted)
plt.show()
print(sklearn.metrics.classification_report(y_test, y_predicted))
y_probas = model.predict_proba(X_test)
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()
predict_final = model.predict(clean_data_x)
predict_final
logistic_submission = pd.DataFrame(
    {'PassengerId': test_clean.PassengerId,
     'Survived': predict_final}).astype('int32')
logistic_submission.head()
logistic_submission.to_csv('Submission.csv',index= False)
