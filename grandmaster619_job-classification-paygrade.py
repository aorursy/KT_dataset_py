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
df = pd.read_csv('../input/job-classification-dataset/jobclassinfo2.csv')

df.info()
df.head(3)
df.isnull().sum()
object_type_features = df.select_dtypes("object").columns

object_type_features
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



for feat_name in object_type_features: 

    df[feat_name] = le.fit_transform(df[feat_name])

df.info()
df.head()
X = df.drop(['PG'], axis = 1)

y = df['PG']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression()



clf_lr.fit(X_train, y_train)

y_pred_lr = clf_lr.predict(X_test)

print("Train Score LR-", clf_lr.score(X_train, y_train)*100 , "%")

print("Test Score LR-", clf_lr.score(X_test, y_test)*100, "%")
from sklearn.metrics import recall_score, precision_score, confusion_matrix



print("Recall score", recall_score(y_test, y_pred_lr, average='macro'))

print("Precision score", precision_score(y_test, y_pred_lr, average='macro'))

print ("CONFUSION MATRIX", confusion_matrix(y_test, y_pred_lr))
from sklearn.tree import DecisionTreeClassifier

clf_dt = DecisionTreeClassifier()



clf_dt.fit(X_train, y_train)

y_pred_dt = clf_dt.predict(X_test)

print("Train Score LR-", clf_dt.score(X_train, y_train)*100 , "%")

print("Test Score LR-", clf_dt.score(X_test, y_test)*100, "%")
print("Recall score", recall_score(y_test, y_pred_dt, average='macro'))

print("Precision score", precision_score(y_test, y_pred_dt, average='macro'))

print ("CONFUSION MATRIX", confusion_matrix(y_test, y_pred_dt))