# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#import dataset
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
# count the number of NaN values in each column
#import dataset
# count the number of NaN values in each column
print(train_df.isnull().sum())
print(test_df.isnull().sum())
# above tells imputation required for three columns: age. cabin, embarked

train_df["Survived"].value_counts()
train_df["Pclass"].value_counts()
train_df["Sex"].value_counts()
train_df["Embarked"].value_counts()

# Taking care of missing data
train_df["Age"] = train_df["Age"].fillna(np.mean(train_df["Age"]))
test_df["Age"] = test_df["Age"].fillna(np.mean(test_df["Age"]))

#covert Sex column to integer
train_df = pd.get_dummies(train_df, prefix="G", columns=["Sex"])
test_df = pd.get_dummies(test_df, prefix="G", columns=["Sex"])

#Deine X & y
X = train_df.iloc[:, [2, 4, 5, 6, 11, 12]].values
y = train_df.iloc[:,1:2].values

#to be used in submission
X_test_df = test_df.iloc[:, [1, 3, 4, 5, 10, 11]].values
print(test_df.isnull().sum())


# Splitting the dataset into the Training set and Test set from train_df itself
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling
#TO DO try diff scaling and normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_test_df = sc.transform(X_test_df)

#-------------------- Logistic Regression ------------------------------
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression()
classifier_lr.fit(X_train, y_train)


# Predicting the Test set results
y_pred_lr = classifier_lr.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm_lr = confusion_matrix(y_test, y_pred_lr)
ac_lr = accuracy_score(y_test, y_pred_lr)
print(classification_report(y_test, y_pred_lr))

#-------------------- KNN ------------------------------
# Fitting KNN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, p = 2)
classifier_knn.fit(X_train, y_train)

# Predicting the Test set results
y_pred_knn = classifier_knn.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm_knn = confusion_matrix(y_test, y_pred_knn)
ac_knn = accuracy_score(y_test, y_pred_knn)
print(classification_report(y_test, y_pred_knn))

#-------------------- Decision Tree ------------------------------
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier()
classifier_dt.fit(X_train, y_train)


# Predicting the Test set results
y_pred_dt = classifier_dt.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm_dt = confusion_matrix(y_test, y_pred_dt)
ac_dt = accuracy_score(y_test, y_pred_dt)
print(classification_report(y_test, y_pred_dt))

#-------------------- Random Forest ------------------------------
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators=100)
classifier_rf.fit(X_train, y_train)


# Predicting the Test set results
y_pred_rf = classifier_rf.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm_rf = confusion_matrix(y_test, y_pred_rf)
ac_rf = accuracy_score(y_test, y_pred_rf)
print(classification_report(y_test, y_pred_rf))

#-------------------- SVM ------------------------------
# Fitting SVM to the Training set
from sklearn.svm import SVC, LinearSVC
classifier_svc = SVC()
classifier_svc.fit(X_train, y_train)

# Predicting the Test set results
y_pred_svc = classifier_knn.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm_svc = confusion_matrix(y_test, y_pred_svc)
ac_svc = accuracy_score(y_test, y_pred_svc)
print(classification_report(y_test, y_pred_svc))

score_dict = {"Random Forest Score": round(ac_rf*100, 2),
              "Decision Tree Score": round(ac_dt*100, 2),
              "KNN Score": round(ac_knn*100, 2),
              "Logistic Regression Score": round(ac_lr*100, 2),
              "SVC": round(ac_svc*100, 2)
        }

#Since rf score is best, using it to derive y_pred for test_df
y_pred_final = classifier_rf.predict(X_test_df)

submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": y_pred_final})
submission.to_csv("Titanic_Baseline_Model_Submission.csv", index=False)

submission.to_csv("Titanic_Baseline_Model_Submission.csv")
