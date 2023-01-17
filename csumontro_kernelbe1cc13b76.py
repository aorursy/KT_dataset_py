# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import confusion_matrix 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
training=pd.read_csv('../input/training.csv')
testing=pd.read_csv('../input/testing.csv')
X_train=training.iloc[:,1:10]
y_train=training.iloc[:,0]
X_test=testing.iloc[:,1:10]
y_test=testing.iloc[:,0]
clf_gini = DecisionTreeClassifier(criterion = "gini", 

            random_state = 100,max_depth=4, min_samples_leaf=9) 
clf_gini.fit(X_train, y_train) 
clf_entropy = DecisionTreeClassifier( 

            criterion = "entropy", random_state = 100, 

            max_depth = 4, min_samples_leaf = 9) 
clf_entropy.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test) 

accuracy_score(y_test,y_pred_gini)*100
classification_report(y_test, y_pred_gini)
y_pred_entropy = clf_entropy.predict(X_test) 

accuracy_score(y_test,y_pred_entropy)*100
classification_report(y_test, y_pred_entropy)