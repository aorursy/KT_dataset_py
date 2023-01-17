# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv(os.path.join(dirname, filename))
#Splitting the dataset into features and target class

X=df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',

       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',

       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]

y=df['Class']
#splitting the features and target class into training and testing datasets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.ensemble import RandomForestClassifier
#initialising the random forest classifier consisting of 20 decision trees

rfc=RandomForestClassifier(n_estimators=20)

rfc.fit(X_train,y_train)
#predicting the target class of testing dataset

y_pred=rfc.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#Obtaining Accuracy Score

print(accuracy_score(y_test,y_pred))
#Obtaining the confusion matrix

print(confusion_matrix(y_test,y_pred))
#Brief summary of the classification

print(classification_report(y_test,y_pred))