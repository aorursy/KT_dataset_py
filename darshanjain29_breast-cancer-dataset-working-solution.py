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
train_df  = pd.read_csv("/kaggle/input/breast-cancer/train.csv")

test_df  = pd.read_csv("/kaggle/input/breast-cancer/test.csv")
test_df.head(3)
X_train = train_df.drop(["Id", "class"], axis = 1)

y_train = train_df['class'] 

X_test = test_df.drop(["Id"], axis = 1)
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression



clf = LogisticRegression() 

clf.fit(X_train, y_train)



Y_pred  = clf.predict(X_test)



scores = cross_val_score(clf, X_train, y_train, cv = 10, scoring = "accuracy")



print ("Scores: ",scores)

print ("Mean: ", scores.mean())

print ("Standard Deviation: ", scores.std())
# 2. SVM

from sklearn import svm

clf_svm = svm.SVC()

clf_svm.fit(X_train, y_train)



Y_pred_svm  = clf_svm.predict(X_test)



scores_svm = cross_val_score(clf_svm, X_train, y_train, cv = 10, scoring = "accuracy")

print ("Scores: ",scores_svm.mean())
from sklearn.ensemble import RandomForestClassifier



clf_rf = RandomForestClassifier(max_depth=2, random_state=0)



clf_rf.fit(X_train, y_train)



Y_pred_rf  = clf_rf.predict(X_test)



scores_rf = cross_val_score(clf_rf, X_train, y_train, cv = 10, scoring = "accuracy")

print ("Scores: ",scores_rf.mean())

output = pd.DataFrame({'Id': test_df['Id'], 'Class': Y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")