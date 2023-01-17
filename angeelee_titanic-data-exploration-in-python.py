import numpy as np

import pandas as pd

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
train_df.tail()
train_df.info()
train_df.describe()
train_df.describe(include=['O'])
train_df.isnull().any()
train_df['Age'].plot(kind='hist')
train_df['Age'].fillna(train_df['Age'].median(),inplace=True)
train_df.drop(['PassengerId','Name','Ticket','Embarked','Cabin'], axis=1,inplace=True)
test_df.isnull().any()
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

test_df['Age'].fillna(test_df['Age'].median(),inplace=True)
test_df.drop(['Name','Ticket','Embarked','Cabin'], axis=1,inplace=True)
import seaborn as sns

sns.barplot(x='Sex', y='Survived', data=train_df)
sns.barplot(x='Pclass', y='Survived', data=train_df)
import matplotlib.pyplot as plt

plt.subplots(figsize=(10,10))

ax = plt.axes()

ax.set_title("Correlation Heatmap")

corr = train_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,cmap='autumn')
import statsmodels.formula.api as sm
lm = sm.ols(formula='Survived~Pclass+Sex+Age+SibSp+Parch+Fare', data=train_df).fit()

lm.summary()
train_df['Sex'].replace(['male','female'],[0,1],inplace=True)

test_df['Sex'].replace(['male','female'],[0,1],inplace=True)
X_train = train_df.drop("Survived",axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred1 = logreg.predict(X_test)

logreg.score(X_train, Y_train)
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred2 = decision_tree.predict(X_test)

decision_tree.score(X_train, Y_train)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier() 

knn.fit(X_train,Y_train)

Y_pred3=knn.predict(X_test)

knn.score(X_train, Y_train)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred1

    })

submission.to_csv('submission.csv', index=False)