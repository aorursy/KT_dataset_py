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
X_test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

X_train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
Y_train=X_train['label']

X_train=X_train.drop(['label'],axis=1)

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
clf=DecisionTreeClassifier()

clf.fit(X_train,Y_train)

ypred=clf.predict(X_test)

acc=round(clf.score(X_train,Y_train)*100,2)

print(acc)
clf1=RandomForestClassifier()

clf1.fit(X_train,Y_train)

ypred1=clf1.predict(X_test)

acc1=round(clf1.score(X_train,Y_train)*100,2)

print(acc1)
submission=pd.DataFrame({"Label": ypred1})





submission.index.name='ImageId'

submission=submission.reset_index()
submission['ImageId']=submission['ImageId']+1

submission.to_csv('submission1.csv', index=False)