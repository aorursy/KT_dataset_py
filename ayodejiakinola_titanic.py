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
df_train=pd.read_csv('/kaggle/input/titanic/train.csv')

df_test=pd.read_csv('/kaggle/input/titanic/test.csv')

df_test0=df_test

df_train.head()
split = df_train.shape[0]

df_tt= pd.concat([df_train, df_test], axis=0)

df_tt.shape
df_tt.info()
df_tt=df_tt.drop(['Cabin'],axis=1)

df_tt['Fare']= df_tt['Fare'].fillna('7.0')

df_tt['Embarked']= df_tt['Embarked'].fillna('S')

#df_tt=df_tt[df_tt.Fare.notnull()]

#df_tt=df_tt[df_tt.Embarked.notnull()]

df_tt['Age']= df_tt['Age'].fillna(df_tt['Age'].mean())
df_tt.info()
df_train = df_tt[:split]

df_test = df_tt[split:]

#df_test0 = df_tt[split:]
import matplotlib.pyplot as plt

table=pd.crosstab(df_train['Pclass'], df_train['Survived'])

table.plot(kind='bar',stacked=True,figsize=(6,6))

plt.show()
table=pd.crosstab(df_train['Sex'], df_train['Survived'])

table.plot(kind='bar',stacked=True,figsize=(6,6))

plt.show()
#df_train['Age_bin']=pd.cut(df_train['Age'],10)

table=pd.crosstab(pd.cut(df_train['Age'],10), df_train['Survived'])

table.plot(kind='bar',stacked=True,figsize=(6,6))

plt.show()
table=pd.crosstab(df_train['SibSp'], df_train['Survived'])

table.plot(kind='bar',stacked=True,figsize=(6,6))

plt.show()
table=pd.crosstab(df_train['Parch'], df_train['Survived'])

table.plot(kind='bar',stacked=True,figsize=(6,6))

plt.show()
table=pd.crosstab(df_train['Embarked'], df_train['Survived'])

table.plot(kind='bar',stacked=True,figsize=(6,6))

plt.show()
df_train=df_train[df_train.Survived.notnull()]
def Digit(value):

    if value == "male":

        return 1

    else:

        return 0

    

df_train['Sex'] = df_train['Sex'].apply(Digit)

df_train['Sex'] = df_train['Sex'].astype(int)
df_train['Embarked'].unique()
df_train.info()
# plot the heatmap and annotation on it

import seaborn as sns

sns.heatmap(df_train.corr(), xticklabels=df_train.columns, yticklabels=df_train.columns, annot=True)
y = pd.DataFrame(df_train['Survived'])

df_train = pd.get_dummies(df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked']])
from sklearn import preprocessing

df_train_column_names = df_train.columns.values

df_train_np = preprocessing.minmax_scale(df_train)

df_train = pd.DataFrame(df_train_np, columns=df_train_column_names)



df_train.head()
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

#from sklearn.model_selection import train_test_split
from sklearn import datasets

from sklearn.model_selection import train_test_split

#from sklearn.metrics import classification_report

from sklearn.svm import SVC
X = pd.DataFrame(df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked_C','Embarked_Q','Embarked_S']])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

clfl=LogisticRegression()

parameters = {"C": [0.0001, 0.001, 0.1, 1, 10, 100]}

fitmodel = GridSearchCV(clfl, param_grid=parameters, cv=5, scoring="accuracy")

fitmodel.fit(X_train, y_train)

fitmodel.best_estimator_, fitmodel.best_params_, fitmodel.best_score_, fitmodel.cv_results_

#fitmodel.grid_scores_
clfl=LogisticRegression(C=fitmodel.best_params_['C'])

clfl.fit(X_train, y_train)

ypred=clfl.predict(X_test)

accuracy_score(ypred, y_test)
   

df_test['Sex'] = df_test['Sex'].apply(Digit)

df_test['Sex'] = df_test['Sex'].astype(int)



df_test = pd.get_dummies(df_test[['Pclass','Sex','Age','SibSp','Parch','Embarked']])



from sklearn import preprocessing

df_test_column_names = df_test.columns.values

df_test_np = preprocessing.minmax_scale(df_test)

df_test = pd.DataFrame(df_test_np, columns=df_test_column_names)



X_test = pd.DataFrame(df_test[['Pclass','Sex','Age','SibSp','Parch','Embarked_C','Embarked_Q','Embarked_S']])
ypred=clfl.predict(X_test)
ypred = ypred.astype('int')
output = pd.DataFrame({'PassengerId': df_test0.PassengerId, 'Survived': ypred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")