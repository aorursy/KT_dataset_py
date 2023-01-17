import seaborn as sns
import pandas as pd

iris = pd.read_csv("../input/iris-data-set-for-beginners/iris.csv")
iris.head()
iris = iris.drop(['Unnamed: 0'],axis=1)

iris.head()
sns.pairplot(iris)
sns.pairplot(iris , hue = "Species")
import matplotlib.pyplot as plt

import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head()
train = train.drop (['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis=1)

train.head()
sns.pairplot(train)   # it's bad for categorical data

plt.show()
sns.catplot(x='Pclass',y='Age',data=train,hue='Sex')
sns.boxplot(x='Pclass',y='Age',data=train,hue='Sex')
train.dropna(axis=0)

train.head()
plt.figure(figsize=(12,8))

sns.distplot(train['Fare'])
sns.jointplot('Age','Fare',data=train)
sns.jointplot('Age','Fare',data=train,kind='kde')
sns.jointplot('Age','Fare',data=train,kind='hex')
sns.heatmap(train.corr())