import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

%matplotlib inline
df = pd.read_csv("../input/titanic123/ChanDarren_RaiTaran_Lab2a.csv")

df.head(5)
sns.countplot(x="Survived", hue="Sex", data=df)
plt.scatter(df.Age,df.Survived)

plt.xlabel("age")

plt.ylabel("Survived")

plt.show()
df.isnull().sum()
df.shape
df.dropna(inplace=True)
df.isnull().sum()
df.drop(["Cabin","Name","PassengerId"], axis=1, inplace=True)
df.head(5)
le_Sex = LabelEncoder()

le_Embarked = LabelEncoder()



df["Sex_n"] = le_Sex.fit_transform(df.Sex)

df["Embarked_n"] = le_Embarked.fit_transform(df.Embarked)
df.head(5)
df.drop(["Sex","Embarked"],axis=1, inplace=True)
df.head(5)
x=df.drop(["Survived"],axis=1)

y=df["Survived"]
x = df.drop(['Ticket'], axis=1)
from sklearn import tree

model = tree.DecisionTreeClassifier()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)





model.fit(x,y)
predic = model.predict(X_test)

accuracy = accuracy_score(y_test, predic)

accuracy
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predic)