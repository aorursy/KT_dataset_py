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
df = pd.read_csv("/kaggle/input/personal-loan/Bank_Personal_Loan_Modelling-1.xlsx")
print(df)
df.head(6)
df.tail(6)
df.describe() # indepth detail of data frame i.e min max values for age feature
df.info()
df.info()
df.drop(["ID"],axis = 1,inplace=True)

X= df.drop(["Personal Loan"],axis=1)
y = df["Personal Loan"]
X.info()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=42)


from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()
clf_gnb.fit(X_train, y_train)

y_pred_gnb = clf_gnb.predict(X_test)
print("Train score",clf_gnb.score(X_train, y_train)*100)
print("Test score",clf_gnb.score(X_test, y_test)*100)
from sklearn.neighbors import KNeighborsClassifier
clf_knn =  KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(X_train, y_train)

y_pred_knn = clf_knn.predict(X_test)
print("Train score",clf_knn.score(X_train, y_train)*100)
print("Test score",clf_knn.score(X_test, y_test)*100)

from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression() 
clf_lr.fit(X_train, y_train)

y_pred_lr = clf_lr.predict(X_test)
print("Train score",clf_lr.score(X_train, y_train)*100)
print("Test score",clf_lr.score(X_test, y_test)*100)