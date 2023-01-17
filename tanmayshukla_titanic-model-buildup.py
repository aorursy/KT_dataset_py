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
data_train=pd.read_csv("/kaggle/input/titanic/train.csv")

data_test=pd.read_csv("/kaggle/input/titanic/test.csv")

data_test.head()

y_train=data_train["Survived"]

# y_test=data_test["Survived"]

features=["Pclass","SibSp","Sex","Parch"]

x_train=pd.get_dummies(data_train[features])

x_test=pd.get_dummies(data_test[features])

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=20,random_state=3,max_depth=5)

clf.fit(x_train,y_train)

# print("Training_result:\n{:.2f}".format(clf.score(x_train,y_train)))

# print("Testing_result:\n{:.2f}".format(clf.score(x_test,y_test)))

predictions=clf.predict(x_test)

output=pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': predictions})

output.to_csv('mysubmission.csv',index=False)

print("Your submission successful\n")