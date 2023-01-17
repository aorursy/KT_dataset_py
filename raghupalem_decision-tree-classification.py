# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
dataset = pd.read_csv("../input/bill_authentication.csv")  

# Any results you write to the current directory are saved as output.
dataset.shape 
#dataset.head
dataset.head()
X=dataset.drop('Class',axis=1)
y=dataset['Class']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
predicted_values=y_pred
predicted_values
my_submission = pd.DataFrame({'Actual_value': y_test[0:], 'predicted_values':predicted_values})

my_submission.to_csv('submission.csv', index=False)
