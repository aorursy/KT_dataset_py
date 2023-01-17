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
import pandas as pd



df_train = pd.read_csv("/kaggle/input/haitidsweek1/train.csv")

df_train
df_test = pd.read_csv("/kaggle/input/haitidsweek1/test.csv")

df_test
x = df_train.drop(['label','id'],axis=1)

y = df_train['label']
test = df_test.drop(['id'],axis=1)

test_id = df_test['id']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, BaggingClassifier

from sklearn import svm

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
model_svm = svm.SVC(kernel='linear')

model_svm.fit(x_train,y_train)

y_svm = model_svm.predict(x_test)

y_svm
print(classification_report(y_test,y_svm,))

print(confusion_matrix(y_test,y_svm))

print(accuracy_score(y_test,y_svm))
model_tree = DecisionTreeClassifier(random_state=1)

model_tree.fit(x_train,y_train)

y_tree = model_tree.predict(x_test)

y_tree
print(classification_report(y_test,y_tree,))

print(confusion_matrix(y_test,y_tree))

print(accuracy_score(y_test,y_tree))
model_bagging = BaggingClassifier(n_estimators=50, random_state=1)

model_bagging.fit(x_train,y_train)

y_bagging = model_bagging.predict(x_test)

y_bagging
print(classification_report(y_test,y_bagging))

print(confusion_matrix(y_test,y_bagging))

print(accuracy_score(y_test,y_bagging))
model_ada = AdaBoostClassifier(n_estimators=100, random_state=1)

model_ada.fit(x_train,y_train)

y_ada = model_ada.predict(x_test)

y_ada
print(classification_report(y_test,y_ada))

print(confusion_matrix(y_test,y_ada))

print(accuracy_score(y_test,y_ada))
model_svm = svm.SVC(kernel='linear')

model_svm.fit(x,y)

y_pred = model_svm.predict(test)

y_pred
submission = pd.DataFrame({'id':test_id,'label':y_pred})

submission
submission.to_csv("submission.csv",index=False)