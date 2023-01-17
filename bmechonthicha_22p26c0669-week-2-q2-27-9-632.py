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
data = pd.read_csv("/kaggle/input/titanic/train.csv")
data.head()
data.isnull().sum()
data = data.drop(['PassengerId','Name','SibSp','Ticket','Age','Cabin','Embarked'], axis = 1)

data.head()
data['Sex'] = data['Sex'].map({'female': 1, 'male': 0})
data.head()
y = data['Survived']

x = data.drop('Survived', axis = 1)

data = pd.concat([x,y], axis=1)
data.head()
kfold = KFold(10, True, 1)
from sklearn.model_selection import cross_val_score

from sklearn import tree

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import recall_score,precision_score,f1_score

from sklearn.neural_network import MLPClassifier

from sklearn import metrics



mean_F1 = []



clf1 = tree.DecisionTreeClassifier()

f_measure = cross_val_score(clf1, x, y, cv=5, scoring='f1_macro')

recall = cross_val_score(clf1, x, y, cv=5, scoring='recall')

precision = cross_val_score(clf1, x, y, cv=5, scoring='precision')



f_measure = sum(f_measure)/len(f_measure)

recall = sum(recall)/len(recall)

precision = sum(precision)/len(precision)

mean_F1.append(f_measure)



print("Decision Tree")

print("F1-score: " ,f_measure)

print("Recall: " ,recall)

print("Precision: " ,precision)





clf1 = GaussianNB()

f_measure = cross_val_score(clf1, x, y, cv=5, scoring='f1_macro')

recall = cross_val_score(clf1, x, y, cv=5, scoring='recall')

precision = cross_val_score(clf1, x, y, cv=5, scoring='precision')



f_measure = sum(f_measure)/len(f_measure)

recall = sum(recall)/len(recall)

precision = sum(precision)/len(precision)

mean_F1.append(f_measure)



print("Decision Tree")

print("F1-score: " ,f_measure)

print("Recall: " ,recall)

print("Precision: " ,precision)





clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

f_measure = cross_val_score(clf1, x, y, cv=5, scoring='f1_macro')

recall = cross_val_score(clf1, x, y, cv=5, scoring='recall')

precision = cross_val_score(clf1, x, y, cv=5, scoring='precision')



f_measure = sum(f_measure)/len(f_measure)

recall = sum(recall)/len(recall)

precision = sum(precision)/len(precision)

mean_F1.append(f_measure)



print("neuron network")

print("F1-score: " ,f_measure)

print("Recall: " ,recall)

print("Precision: " ,precision)





print("Mean F1-score: ", sum(mean_F1)/len(mean_F1))