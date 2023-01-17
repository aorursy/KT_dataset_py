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
import numpy as np

import pandas as pd
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

gsub = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
df_train.info()
df_train.head
df_train.isnull().sum()
df_test.isnull().sum()
gsub.isnull().sum()
df_train=df_train.drop(["Cabin"],axis=1)
df_test=df_test.drop(["Cabin"],axis=1)
df_train['Age'].fillna(df_train['Age'].median(), inplace= True)

df_train['Embarked'].fillna(df_train['Embarked'].mode(), inplace= True)
df_test['Age'].fillna(df_test['Age'].median(), inplace= True)

df_train['Embarked'].fillna(df_train['Embarked'].mode(), inplace= True)
df_train = df_train.drop(["PassengerId", "Fare", "Ticket", "Name"], axis = 1)
PassengerId = df_test["PassengerId"]

df_test = df_test.drop(["PassengerId", "Fare", "Ticket", "Name"], axis = 1)
from sklearn.preprocessing import LabelEncoder
c= df_train.drop(df_train.select_dtypes(exclude=['object']), axis=1).columns

print(c)

e1 = LabelEncoder()

df_train[c[0]] = e1.fit_transform(df_train[c[0]].astype('str'))

e2 = LabelEncoder()

df_train[c[1]] = e2.fit_transform(df_train[c[1]].astype('str'))
df_test[c[0]] = e1.transform(df_test[c[0]].astype('str'))

df_test[c[1]] = e2.transform(df_test[c[1]].astype('str'))
import matplotlib.pyplot as plt

import seaborn as sns

fig, ax=plt.subplots(figsize=(12,10))

corr = df_train.corr()

sns.heatmap(corr, annot= True,cmap='Set1',linewidth=1.0)
X = df_train.drop(['Survived'], axis=1)

y = df_train['Survived']
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
from sklearn.linear_model import LogisticRegression

L_model = LogisticRegression()

L_model.fit(X_train, y_train)
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
y_pred = L_model.predict(X_test)

pred= pd.DataFrame({'Actual_values': y_test, 'Predicted_values': y_pred})

pred.head()
from sklearn import metrics

from sklearn.metrics import mean_absolute_error,mean_squared_error
print("Accuracy score : {0}".format(metrics.accuracy_score(y_pred, y_test)))
m_sq_err=mean_squared_error(y_test,y_pred)

print("mean_squared_error is {}".format(m_sq_err))
m_abs_err=mean_absolute_error(y_test,y_pred)

print("mean_absolute_error is {}".format(m_abs_err))
y_p_test = L_model.predict(df_test)

y_p_test
submission = pd.DataFrame({

"PassengerId": PassengerId,

"Survived": y_p_test

})

submission.to_csv('./submission.csv', index= False)