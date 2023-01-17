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
data_master = pd.read_csv('/kaggle/input/personal-loan/Bank_Personal_Loan_Modelling-1.xlsx')

data_master.shape # No.of Rows & Columns
data_master.head(2)
data_master.describe()
data_master.info() #Step 3 & 4
# Step 3

data_master.isnull().sum()
#Step 5 - Create X & Y

data_master.drop("ID",axis=1, inplace=True)
X=data_master.drop("Personal Loan", axis=1)

y=data_master["Personal Loan"]
X.info()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)

#1 GaussianNB Classifier

from sklearn.naive_bayes import GaussianNB

clf_gnb = GaussianNB()

clf_gnb.fit(X_train, y_train)

y_pred_gnb=clf_gnb.predict(X_test)

print("Train Score - ",clf_gnb.score(X_train,y_train)*100)

print("Test Score - ",clf_gnb.score(X_test,y_test)*100)

#2 KNN Classifier

from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier(n_neighbors=3)

clf_knn.fit(X_train, y_train)

y_pred_knn=clf_knn.predict(X_test)

print("Train Score - ",clf_knn.score(X_train,y_train)*100)

print("Test Score - ",clf_knn.score(X_test,y_test)*100)



#3 SVM Classifier

from sklearn import svm

clf_svm = svm.SVC()

clf_svm.fit(X_train, y_train)

y_pred_svm=clf_svm.predict(X_test)

print("Train Score - ",clf_svm.score(X_train,y_train)*100)

print("Test Score - ",clf_svm.score(X_test,y_test)*100)
#4 Logistic Regression Classifier

from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression(random_state=0)

clf_lr.fit(X_train, y_train)

y_pred_lr=clf_lr.predict(X_test)

print("Train Score - ",clf_lr.score(X_train,y_train)*100)

print("Test Score - ",clf_lr.score(X_test,y_test)*100)