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
import matplotlib.pyplot as plt

from sklearn import datasets,linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

from sklearn import neighbors,svm

from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn import svm

import seaborn as sns

from collections import Counter
train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")

print(test_data.head())
train_data.drop(['Cabin','Name','SibSp'],axis=1,inplace=True)



train_data.head()
test_data.drop(['Cabin','Name','SibSp'],axis=1,inplace=True)

test_data.head()
train_data.head()
sns.barplot(x=train_data['Survived'],y=train_data['Age'])
sns.barplot(x=train_data['Survived'],y=train_data['Pclass'])
sns.barplot(x=train_data['Survived'],y=train_data['Parch'])
sns.barplot(x=train_data['Survived'],y=train_data['Fare'])
sns.lmplot(x="PassengerId", y="Age", hue="Survived", data=train_data)
sns.lmplot(x="PassengerId", y="Fare", hue="Survived", data=train_data)
train_data.info()

test_data.info()
train_data.dropna(inplace = True)

train_data.info()
one_hot_encoded_train_data = pd.get_dummies(train_data)

train_data.info()
one_hot_encoded_test_data = pd.get_dummies(test_data)

final_train , final_test = one_hot_encoded_train_data.align(one_hot_encoded_test_data,join='left', axis=1)
final_test.fillna(0,inplace=True)
final_test.head()
ft = ['Pclass', 'Sex_female','Sex_male', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S','Age','Parch']

X_train=final_train[ft]

y_train = final_train['Survived']

X_test=final_test[ft]
X_train.head()
clf = VotingClassifier([('lsvc',svm.LinearSVC()),('knn',neighbors.KNeighborsClassifier()),('rfor',RandomForestClassifier())])

clf.fit(X_train,y_train)
confidence = clf.score(X_train,y_train)

print('Accuracy',confidence)
predictions = clf.predict(X_test)

print('Predicted spread:',Counter(predictions))
df = pd.DataFrame({'PassengerId':test_data['PassengerId']})
len(df)
df['Survived'] = predictions
df.head()

df.info()
# df.set_index('PassengerId',inplace = True)

df.head()
df.to_csv(r"C:\Users\Admin\Desktop\final.csv",index = False)