import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

test_full=test.copy(deep=True)

train.head()
train.info()
train.drop(columns=['Cabin','Name','Ticket','PassengerId'],inplace=True)

test.drop(columns=['Cabin','Name','Ticket','PassengerId'],inplace=True)



train.info()
test.info()
train.Age.fillna(train.Age.median(),inplace=True)

test.Age.fillna(test.Age.median(),inplace=True)



test.Fare.fillna(test.Age.median(),inplace=True)



train.Embarked.fillna(train.Embarked.mode()[0],inplace=True)

test.Embarked.fillna(test.Embarked.mode()[0],inplace=True)
labeler = LabelEncoder()



train['Sex']=labeler.fit_transform(train['Sex'])

train['Embarked']=labeler.fit_transform(train['Embarked'])



test['Sex']=labeler.fit_transform(test['Sex'])

test['Embarked']=labeler.fit_transform(test['Embarked'])
y_train = train['Survived']

X_train = train.drop(columns='Survived')
plt.figure(figsize=(10,6))

corr = train.corr()

sns.heatmap(abs(corr),cmap='Blues',annot=True)
vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif["features"] = X_train.columns

vif
scale = StandardScaler()

X_train = scale.fit_transform(X_train)
lr = linear_model.LogisticRegression()

lr.fit(X_train,y_train)

lr.score(X_train,y_train)
predictions = lr.predict(test)

output=pd.DataFrame({'PassengerId': test_full.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)