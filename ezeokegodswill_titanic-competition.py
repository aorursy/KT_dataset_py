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
!ls
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_train.head()
df_train.isna().sum()
df_train.Embarked.value_counts()
df_test.Embarked.value_counts()
# fillna for embarked
df_train.Embarked.fillna('Q', inplace=True)
df_test.Embarked.fillna('Q', inplace=True)
# fillna for cabin
df_train.Cabin.fillna('Cabin', inplace=True)
df_test.Cabin.fillna('Cabin', inplace=True)
# fillna for age
df_train.Age.fillna(df_train.Age.mean(), inplace=True)
df_test.Age.fillna(df_test.Age.mean(), inplace=True)
df_train.isna().sum()
# unimportant Name and Ticket
unimportant = ["Name", "Ticket"]
# dropping the unimportant columns
df_train.drop(columns=[i for i in unimportant], inplace=True)
df_test.drop(columns=[i for i in unimportant], inplace=True)
df_train.Parch.value_counts()
df_train.head()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

# encode sex
df_train.Sex = label_encoder.fit_transform(df_train.Sex)
df_test.Sex = label_encoder.fit_transform(df_test.Sex)

# encode Embarked
df_train.Embarked = label_encoder.fit_transform(df_train.Embarked)
df_test.Embarked = label_encoder.fit_transform(df_test.Embarked)

# encode Cabin
df_train.Cabin = label_encoder.fit_transform(df_train.Cabin)
df_test.Cabin = label_encoder.fit_transform(df_test.Cabin)
X = df_train.loc[:, df_train.columns != 'Survived']
y = df_train.Survived
X = X.drop("PassengerId", axis=1)
X.head()
# standardize our X data
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

skfolds = StratifiedKFold(n_splits=10)
xgc_clf = XGBClassifier(max_depth = 6, n_estimators=500, learning_rate=0.01)

for train_index, test_index in skfolds.split(X, y):
    X_train_folds = X[train_index]
    y_train_folds = (y[train_index])
    X_test_fold = X[test_index]
    y_test_fold = (y[test_index])
    xgc_clf.fit(X_train_folds, y_train_folds)
    y_pred = xgc_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print("Results for xgboost classifier", n_correct / len(y_pred))
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
xgc_y_pred = cross_val_predict(xgc_clf, X, y, cv=10)
confusion_matrix(y, xgc_y_pred)
# #Precision for our xgc_clf classifier
from sklearn.metrics import precision_score, recall_score
print('The precision is :', precision_score(y, xgc_y_pred))
print('The recall is :', recall_score(y, xgc_y_pred))
df_test.head()
df_test = df_test.drop("PassengerId", axis=1)
df_test = scaler.fit_transform(df_test)
pred = xgc_clf.predict(df_test)
predDf = pd.DataFrame(pred, columns=["Survived"])
predDf.head()
predDf.Survived.value_counts()
df_train.dtypes
pred_test = pd.read_csv("test.csv")
pred_test
%cd working/
submission = pd.DataFrame({'PassengerId':pred_test['PassengerId'], 'Survived':predDf["Survived"]})

filename = "First_submission.csv"
submission.to_csv(filename, index=False)
print("Saved file: ", filename)
