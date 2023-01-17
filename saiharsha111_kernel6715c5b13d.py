import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/titanic/train.csv")

df.head()
df.drop(['PassengerId','Cabin','Name','Ticket'],axis=1,inplace=True)

df.head()
sns.countplot(df["Survived"],hue=df["Sex"])
sns.countplot(df["Survived"],hue=df["Embarked"])
sns.countplot(df["Survived"],hue=df["Pclass"])
df.isnull().sum()
df = df[df["Embarked"].notna()]

df.isnull().sum()
sns.boxplot(x="Pclass",y="Age",data=df)
def find_age(col):

    age = col[0]

    Pclass = col[1]

    if pd.isnull(age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return age
df["Age"] = df[["Age","Pclass"]].apply(find_age,axis=1)

df.head()
plt.hist(df["Age"])
df.isnull().sum()
sns.heatmap(df.corr(),annot=True)
bins= [0,10,20,30,40,50,60,70,80,90]

labels = ['0','1','2','3','4','5','6','7','8']

df['Agegroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

df.head()
df["Embarked"].value_counts()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(["S","C","Q"])

df["Embarked"] = le.fit_transform(df["Embarked"])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
s = scaler.fit(df[["Fare"]])
df["Fare"] = s.transform(df[["Fare"]])

df.head()
df["Male"] = pd.get_dummies(df["Sex"],drop_first=True)
df.drop("Sex",inplace=True,axis=1)
df.columns
x = df[['Male','Agegroup','SibSp','Pclass', 'Parch', 'Fare', 'Embarked']].values

y = df["Survived"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.35)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=4,random_state=10)
tree.fit(x_train,y_train)
predict = tree.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predict)
test = pd.read_csv("../input/titanic/test.csv")

test.head()
test.drop(['Cabin','Name','Ticket'],axis=1,inplace=True)
test.isnull().sum()
def find_age(col):

    age = col[0]

    Pclass = col[1]

    if pd.isnull(age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return age
test["Age"] = test[["Age","Pclass"]].apply(find_age,axis=1)

df.head()
test = test.fillna(df.mean())

test.isnull().sum()
test.shape
test.columns
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(["S","C","Q"])

test["Embarked"] = le.fit_transform(test["Embarked"])
le.fit(["male","female"])

test["Sex"] = le.fit_transform(test["Sex"])

test["Male"] = pd.get_dummies(test["Sex"],drop_first=True)

test.head()
scaler = StandardScaler()
s = scaler.fit(test[["Fare"]])
test["Fare"] = s.transform(test[["Fare"]])

test.head()
bins= [0,10,20,30,40,50,60,70,80,90]

labels = ['0','1','2','3','4','5','6','7','8']

test['Agegroup'] = pd.cut(test['Age'], bins=bins, labels=labels, right=False)

test.head()
x = test[['Male', 'Agegroup', 'SibSp','Pclass', 'Parch', 'Fare', 'Embarked']].values
ypredict = tree.predict(x)

ypredict
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':ypredict})
submission.head()
filename = 'Titanic1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
submission.shape