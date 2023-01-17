# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

# print(train_data.head())

y_train = train_data["Survived"]

print(y_train)

x_train = train_data.drop(["Survived"],axis=1)

print(x_train)
from sklearn.linear_model import LinearRegression
x_train.head()

x_train = train_data.drop(["Name","Ticket","Cabin"],axis=1)

x_train.head()
# x_train['Sex_s'] = x_train['Sex'].replace(0, 'Female',inplace=True)

# x_train['Sex_s']= x_train['Sex'].replace(1, 'Male',inplace=True)

gender = {'male': 1,'female': 2} 

x_train.Sex = [gender[item] for item in x_train.Sex] 

x_train = x_train.drop(["Embarked"],axis = 1)

x_train.head()
x_train = x_train.dropna(axis=1)

clf = LinearRegression()

clf.fit(x_train,y_train)
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

# print(train_data.head())

# print(x_test)

# print(x_test)

x_test = train_data.drop(["Name","Ticket","Cabin"],axis=1)

gender = {'male': 1,'female': 2} 

x_test.Sex = [gender[item] for item in x_test.Sex] 

x_test = x_test.drop(["Embarked"],axis = 1)

x_test = x_test.dropna(axis=1)

x_test.head()

y_pred = clf.predict(x_test)
list(y_pred).index(max(list(y_pred)))
x_train.describe()

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.isnull().sum()

train = pd.read_csv("/kaggle/input/titanic/train.csv")



for col in train.columns: 

    print(train.groupby(col).Survived.value_counts())
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(x_train, y_train)

y_pred_log_reg = clf.predict(x_test)

acc_log_reg = round( clf.score(x_train, y_train) * 100, 2)

print (str(acc_log_reg) + ' percent')
from sklearn.svm import SVC, LinearSVC

clf = SVC()

clf.fit(x_train, y_train)

y_pred_svc = clf.predict(x_test)

acc_svc = round(clf.score(x_train, y_train) * 100, 2)

print (acc_svc)