import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
sns.set()
df = pd.read_csv('../input/titanic/train.csv')
df.head()
df.shape
df.isnull().sum()
df = df.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)
df = df.dropna()
df['Family Size'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = 0

df['IsAlone'].loc[df['Family Size']>1] = 1
df_dummies = pd.get_dummies(df,drop_first=True)
df_dummies.head()
cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male',

       'Embarked_Q', 'Embarked_S','Family Size','IsAlone']
sns.distplot(df_dummies['Age'])
sns.distplot(df_dummies['Fare'])
inputs = df_dummies[cols]
targets = df_dummies['Survived']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
inputs_scaled = scaler.fit_transform(inputs)
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(inputs_scaled,targets,test_size=0.2,random_state=44)
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression(random_state=365)
reg.fit(X_train,y_train)
pred = reg.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100,random_state=365) 

model.fit(X_train,y_train)
predictions = model.predict(X_test)
accuracy_score(predictions,y_test)
from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train,y_train)
svm_pred = svm_model.predict(X_test)
accuracy_score(svm_pred,y_test)
test = pd.read_csv('../input/titanic/test.csv')

test2 = test.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)

test2 = test2.fillna(0)
test2['Family Size'] = test2['SibSp'] + test2['Parch'] + 1
test2['IsAlone'] = 0

test2['IsAlone'].loc[test2['Family Size']>1] = 1
test_dummies = pd.get_dummies(test2,drop_first=True)
test_dummies = test_dummies[cols]
inputs_scaled = scaler.fit_transform(test_dummies)
predictions_final = model.predict(inputs_scaled)
predictions_final
submissions = pd.DataFrame(predictions_final, columns=['Survived'])
submissions['PassengerId']=test['PassengerId']
submissions = submissions[['PassengerId','Survived']]
submissions.to_csv("submissions.csv",index=False)