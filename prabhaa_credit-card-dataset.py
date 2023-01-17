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
credit_card_data = pd.read_csv('../input/creditcard.csv')
credit_card_data.head(5)
credit_card_data.describe()
#print(credit_card_data.dtypes)
#credit_card_data.isnull().any()
print("No Frauds",round(credit_card_data['Class'].value_counts()[0]/len(credit_card_data)*100,2),"%")
print("Frauds",round(credit_card_data['Class'].value_counts()[1]/len(credit_card_data)*100,2),"%")
features = credit_card_data.iloc[:,0:-1]
labels = credit_card_data.iloc[:,-1]
print(features.shape)
print(labels.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.2,random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn import tree
decision_tree_model = tree.DecisionTreeClassifier()
decision_tree_model.fit(x_train,y_train)
y_pred = decision_tree_model.predict(x_test)
from sklearn.metrics import accuracy_score
round(accuracy_score(y_test,y_pred)*100,2)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))