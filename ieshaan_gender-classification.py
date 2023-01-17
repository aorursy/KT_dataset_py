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
df=pd.read_csv('../input/gender-classification/Transformed Data Set - Sheet1.csv')

df.head()
df.isna().sum()
print(df['Favorite Color'].unique())

print(df['Favorite Music Genre'].unique())

print(df['Favorite Beverage'].unique())

print(df['Favorite Soft Drink'].unique())
# Data pre processing



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

lb = LabelEncoder()

df['Gender'] = lb.fit_transform(df['Gender'])
df
X = df.drop(columns=['Gender'],axis=1)

Y = df['Gender']
X = pd.get_dummies(X)
X
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
Accuracy_score=list()
# Logistic Regression



from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(X_train,Y_train)

y_pred=lr.predict(X_test)



from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test, y_pred))

Accuracy_score.append(accuracy_score(Y_test, y_pred))
# Naive Bayes



from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,Y_train)

y_pred1=nb.predict(X_test)



from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test, y_pred1))

Accuracy_score.append(accuracy_score(Y_test, y_pred1))
# Stochastic Gradient Descent



from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss='modified_huber',shuffle=True,random_state=101)

sgd.fit(X_train,Y_train)

y_pred2=sgd.predict(X_test)



from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test, y_pred2))

Accuracy_score.append(accuracy_score(Y_test, y_pred2))
# SVM



from sklearn.svm import SVC

svm = SVC(kernel='linear',C=0.025,random_state=101)

svm.fit(X_train,Y_train)

y_pred3=svm.predict(X_test)



from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test, y_pred3))

Accuracy_score.append(accuracy_score(Y_test, y_pred3))