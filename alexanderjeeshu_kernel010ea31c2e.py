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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
# Importing Decission Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# Importing Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
# Importing Adaboost classifier
from sklearn.ensemble import AdaBoostClassifier
# Importing KNN classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# splitting the train data into test train sets 
train_t,test_t = train_test_split(train,test_size = 0.3, random_state =100)
# defining or test_x ,test_y ,train_x and train_y
train_y = train_t['label']
test_y = test_t['label']
train_x = train_t.drop('label', axis=1)
test_x = test_t.drop('label', axis=1)
#Decission Tree
model_dtc = DecisionTreeClassifier(random_state=100)   
model_dtc.fit(train_x,train_y)

test_pred_dtc = model_dtc.predict(test_x)
acc = accuracy_score(test_y, test_pred_dtc)
print(acc)
#Random Forest
model_rf = RandomForestClassifier(random_state=100, n_estimators=1000)
model_rf.fit(train_x,train_y)

test_pred_rf = model_rf.predict(test_x)
acc1 = accuracy_score(test_y, test_pred_rf)
print(acc1)
#Adaboost
model_ada = AdaBoostClassifier(random_state=100, n_estimators=1000)
model_ada.fit(train_x,train_y)

test_pred_ada = model_ada.predict(test_x)
acc2 = accuracy_score(test_y, test_pred_ada)
print(acc2)
#KNN
#model_knn = KNeighborsClassifier(n_neighbors=5)
#model_knn.fit(train_x,train_y)

#test_pred_knn = model_knn.predict(test_x)
#acc3 = accuracy_score(test_y, test_pred_knn)
#print(acc3)
#Applying RandomForest as its accuracy is better
test_pred = model_rf.predict(test)
df_test_pred = pd.DataFrame(test_pred, columns=['label'])
df_test_pred['ImageId'] = test.index+1
df_test_pred[['ImageId', 'label']].to_csv('submission.csv', index=False)
df_test_pred['ImageId'].shape
