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
df=pd.read_csv('../input/financial-distress/Financial Distress.csv')
df
df.info()
Y=df['Financial Distress']

Y
for y in range(0,len(Y)): # Coverting the values into binary

       if Y[y] > -0.5:

              Y[y] = 0

       else:

              Y[y] = 1
Y.unique()
X=df.drop(columns=['Financial Distress'])

X
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
#Logistic Regression

from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression()

classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_pred,Y_test)

print("Confusiion Matrix is :")

print(cm)

from sklearn.metrics import accuracy_score

print("Accuracy Score is : ", accuracy_score(Y_pred,Y_test))
#Support Vector Machine

from sklearn.svm import SVC

classifier=SVC(kernel='rbf')

classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_pred,Y_test)

print("Confusiion Matrix is :")

print(cm)

from sklearn.metrics import accuracy_score

print("Accuracy Score is : ", accuracy_score(Y_pred,Y_test))
#XGBoost CLassification

from xgboost import XGBClassifier

classifier=XGBClassifier()

classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_pred,Y_test)

print("Confusiion Matrix is :")

print(cm)

from sklearn.metrics import accuracy_score

print("Accuracy Score is : ", accuracy_score(Y_pred,Y_test))