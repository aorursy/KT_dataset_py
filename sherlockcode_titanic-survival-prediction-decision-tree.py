import sklearn as ski

import pandas as pd

import numpy as np

import seaborn as sea

import matplotlib as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

df_test = pd.read_csv("../input/titanic/test.csv")



df_train = pd.read_csv("../input/titanic/train.csv")
df_train.dtypes
df_train.shape
df_train.head(5)

df_train['PassengerId'] = df_train['PassengerId'].replace(np.nan, 0)

df_train['Survived'] = df_train['Survived'].replace(np.nan, 0)

df_train['Pclass'] = df_train['Pclass'].replace(np.nan, 0)

df_train['Name'] = df_train['Name'].replace(np.nan, 0)

df_train['Sex'] = df_train['Sex'].replace(np.nan, 0)

df_train['Age'] = df_train['Age'].replace(np.nan, 0)

df_train['SibSp'] = df_train['SibSp'].replace(np.nan, 0)

df_train['Parch'] = df_train['Parch'].replace(np.nan, 0)

df_train['Ticket'] = df_train['Ticket'].replace(np.nan, 0)

df_train['Fare'] = df_train['Fare'].replace(np.nan, 0)

df_train['Cabin'] = df_train['Cabin'].replace(np.nan, 0)

df_train['Embarked'] = df_train['Embarked'].replace(np.nan, 0)

df_train.head(5)
df_dummies =  pd.get_dummies(df_train, columns=["Sex","Embarked"])

df_dummies.head(5)
df_dummies.drop(['Cabin','Ticket','Name','PassengerId','SibSp'], axis=1,inplace=True)

df_dummies.head(5)
pearson_coefficeint = df_dummies.corr(method='pearson')

pearson_coefficeint
plt.pyplot.figure(figsize=(20,15))

sea.heatmap(pearson_coefficeint, cmap='RdBu_r',annot=True)
x = df_dummies[['Sex_female','Sex_male']]

y = df_dummies['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.8)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
survived_tree = DecisionTreeClassifier(criterion= "entropy", max_depth= 2)

survived_tree
survived_tree.fit(x_train,y_train)
pred_survived = survived_tree.predict(x_test)

pred_survived
from sklearn import metrics

print("Accuracy",metrics.accuracy_score(y_test,pred_survived))