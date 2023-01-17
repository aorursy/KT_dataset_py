import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
import os
os.path.realpath('.')
#train=pd.read_csv("..")
train=pd.read_csv("/Users/radhikasriv/Downloads/train.csv")
test=pd.read_csv("/Users/radhikasriv/Downloads/test.csv")
train.head()
test.head()
train.describe
train.columns
train[['Pclass','Survived']].head()
train['Survived'].count()
train.dtypes
pd.isna(train).sum()
#shows how many are missing in each column
train["Age"]
train['Embarked']=train['Embarked'].fillna('S')
test['Embarked']=test['Embarked'].fillna('S')
pd.isna(train).sum()
pd.isna(test).sum()
train = train.drop('Cabin', axis=1)
test = test.drop('Cabin', axis = 1)
pd.isna(train).sum()
pd.isna(test).sum()
train=train.drop('Age', axis=1)
test=test.drop('Age', axis=1)
pd.isna(test).sum()
pd.isna(train).sum()
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
pd.isna(test).sum()
test.head()
passengerid = test['PassengerId']
train.head()
Y_train = train["Survived"]
X_train = train.drop(['Survived',"PassengerId","Name","Ticket", "Embarked","Sex"],axis=1)
test = test.drop(["PassengerId","Name","Ticket","Embarked","Sex"],axis=1)
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
pred = model.predict(test)
#accuracy_score(Y_train, pred)
pred[0:5]
submission = pd.DataFrame({
      "PassengerId": passengerid,
      "Survived": pred
  })

submission.PassengerId = submission.PassengerId.astype(int)
submission.Survived = submission.Survived.astype(int)

submission.to_csv("titanic1_submission.csv", index=False)
submission
