# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
mushrooms = pd.read_csv('../input/mushrooms.csv')
mushrooms.head()

mushrooms.info()
mushrooms.isnull().sum()
mr = mushrooms
def is_edible(value):
    return 1 if value == 'e' else 0
outcome=mushrooms['class'].apply(is_edible)
outcome.describe()
outcome.value_counts()
# for column in mr:
#     print(mr[column].value_counts())


sns.countplot(x='class',data=mushrooms)
columns=mushrooms.columns[1:23]
plt.subplots(figsize=(22,20))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    mushrooms[i].hist(bins=20,edgecolor='black')
    plt.title(i)
plt.show()
edi = mushrooms[mushrooms['class']=='p']
columns=mushrooms.columns[1:23]
plt.subplots(figsize=(22,20))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    edi[i].hist(bins=20,edgecolor='black')
    plt.title(i)
plt.show()
sns.pairplot(data=mushrooms)
plt.show()
# mushrooms
### good run
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
# outcome=mushrooms['class']
# data=mushrooms[mushroom.columns[:23]]
train,test=train_test_split(mushrooms,test_size=0.25,random_state=0,stratify=mushrooms['class'])# stratify the outcome
# train_X=train[['bruises','odor']]
train_X=train[['bruises','odor','gill-color','gill-size','stalk-surface-below-ring','veil-color','ring-number','ring-type','spore-print-color']]
one_hot_encoded_train_X = pd.get_dummies(train_X)
one_hot_encoded_train_X.head()
# test_X=test[['bruises','odor']]
test_X=test[['bruises','odor','gill-color','gill-size','stalk-surface-below-ring','veil-color','ring-number','ring-type','spore-print-color']]
one_hot_encoded_test_X = pd.get_dummies(test_X)
one_hot_encoded_test_X.head()
train_Y=train['class'].apply(is_edible)
# train_Y=train['class']
test_Y=test['class'].apply(is_edible)

# ###  ** try with bad data 
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import KFold
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
# import warnings
# warnings.filterwarnings('ignore')
# # outcome=mushrooms['class']
# # data=mushrooms[mushroom.columns[1:23]]
# train,test=train_test_split(mushrooms,test_size=0.25,random_state=0,stratify=mushrooms['class'])# stratify the outcome
# # train_X=train[['bruises','odor']]
# # train_X=train[['bruises','odor','gill-color','gill-size','stalk-surface-below-ring','veil-color','ring-number','ring-type','spore-print-color']]
# train_X = train[['cap-shape','cap-surface']]
# one_hot_encoded_train_X = pd.get_dummies(train_X)
# one_hot_encoded_train_X.head()
# # test_X=test[['bruises','odor']]
# # test_X=test[['bruises','odor','gill-color','gill-size','stalk-surface-below-ring','veil-color','ring-number','ring-type','spore-print-color']]
# # test_X = test[test.columns[1:23]] # test with all columsn
# test_X = test[['cap-shape','cap-surface']]
# one_hot_encoded_test_X = pd.get_dummies(test_X)
# one_hot_encoded_test_X.head()
# train_Y=train['class'].apply(is_edible)
# # train_Y=train['class']
# test_Y=test['class'].apply(is_edible)
# train_X.head()
# # one_hot_encoded_train_X.head()
# train[train.columns[1:23]].head()
one_hot_encoded_train_X.head()
# train_Y.head()


train_Y.head()
mushrooms.info()
mushrooms.head(2)

model = LogisticRegression()
model.fit(one_hot_encoded_train_X,train_Y)
prediction=model.predict(one_hot_encoded_test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_Y))

print(model.predict(one_hot_encoded_test_X.head(5)))
one_hot_encoded_test_X.head(5)
mushrooms['class'][1170]
# importing alll the necessary packages to use the various classification algorithms
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.cross_validation import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
model=DecisionTreeClassifier()
model.fit(one_hot_encoded_train_X,train_Y)
prediction=model.predict(one_hot_encoded_test_X)
# model.fit(train_X,train_Y) 
# prediction=model.predict(test_X) 
print('The accuracy of the Decision Tree using Petals is:',metrics.accuracy_score(prediction,test_Y))
