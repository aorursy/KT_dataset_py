# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

submiss=pd.read_csv("../input/sample_submission.csv")
# Any results you write to the current directory are saved as output.
submiss.head()
test.head()
train_data.head()
train_data.shape
from sklearn.preprocessing import LabelEncoder
def encode(train, test):
    le = LabelEncoder().fit(train.species) 
    labels = le.transform(train.species)           # encode species strings
    classes = list(le.classes_) 
    test_ids = test.id 
    return labels,test_ids, classes

labels,test_ids, classes = encode(train_data, test_data)
train_data['species'].value_counts()
# Check Any Null Value
#train_data.isnull().sum()
#import pandas_profiling as pf
#pf.ProfileReport(train_data)
# Import Some Library

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import auc, roc_curve
train, test = train_test_split(train_data, test_size = .33, random_state = 100)

train_y = train['species']
test_y = test['species']

train_x = train.drop('species', axis=1)
test_x = test.drop('species', axis=1)

print(train_y.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# Decision Tree

dec = DecisionTreeClassifier()
dec.fit(train_x,train_y)

pred_dec = dec.predict(test_x)

dec_accuracy = accuracy_score(pred_dec, test_y)
dec_class_report = classification_report(pred_dec,test_y)

print(dec_accuracy)
print(dec_class_report)
# Random Forest
rf = RandomForestClassifier(random_state=100, n_estimators=1000)
rf.fit(train_x, train_y)
rf_pred = rf.predict(test_x)
rf_accuracy = accuracy_score(test_y, rf_pred)
rf_class = classification_report(test_y, rf_pred)

print(rf_accuracy)
print(rf_class)
# KNN
#knn = KNeighborsClassifier()
#knn.fit(train_x,train_y)

#knn_pred = knn.predict(test_x)
#knn_accuracy = accuracy_score(test_y,knn_pred)
#knn_class = classification_report(test_y, knn_pred)
#print(knn_accuracy)
#print(knn_class)
# Naviyes_byes
br = BernoulliNB()
br.fit(train_x, train_y)
pred_br = br.predict(test_x)
accuracy_br = accuracy_score(test_y, pred_br)
class_br = classification_report(test_y,pred_br)


#
ga = GaussianNB()
ga.fit(train_x, train_y)
pred_ga = ga.predict(test_x)
accuracy_ga = accuracy_score(test_y, pred_ga)
class_ga = classification_report(test_y,pred_ga)

print(accuracy_br,accuracy_ga)
ada = AdaBoostClassifier()
ada.fit(train_x, train_y)
pred_ada = ada.predict(test_x)
accuracy_ada = accuracy_score(test_y, pred_ada)
class_ada = classification_report(test_y,pred_ada)
print(accuracy_ada,class_ada)
## We use this data RandomForest

rf = RandomForestClassifier(random_state=100, n_estimators=1000)
rf.fit(train_x,train_y)
pred_rf = rf.predict_proba(test_data)

classes
submission = pd.DataFrame(pred_rf,columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()

submission=submission.drop('index',axis=1)
submission.to_csv("Submission.csv",index=False)