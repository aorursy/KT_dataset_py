import pandas as pd 

import numpy as np
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.head()
train.drop(['Name'],axis=1,inplace=True)
train.head()
train.columns
train.index
sample_sub=pd.read_csv("../input/gender_submission.csv")
sample_sub.head()
train.head()
test.head()
test.drop(['Name'],axis=1,inplace=True)
test.head()
train.isnull().sum()
train.index
test.isnull().sum()
train.drop(['Cabin'],axis=1,inplace=True)

test.drop(['Cabin'],axis=1,inplace=True)

test.drop(['Ticket'],axis=1,inplace=True)

train.drop(['Ticket'],axis=1,inplace=True)

train.drop(['PassengerId'],axis=1,inplace=True)

test.drop(['PassengerId'],axis=1,inplace=True)
test.head()
train.head()
import seaborn as sns

sns.countplot(x='Survived',hue='Sex',data=train)
sns.countplot(x='Survived',hue='Parch',data=train)
sns.countplot(x='Survived',hue='SibSp',data=train)
sns.countplot(x='Survived',hue='Pclass',data=train)
train.isnull().sum()
test.isnull().sum()
train['Age'].mean()
train['Age'].fillna((train['Age'].mean()), inplace=True)
test['Age'].fillna((test['Age'].mean()), inplace=True)
test['Fare'].fillna((test['Fare'].mean()), inplace=True)
train.dropna()
test.isnull().sum()
train.columns
train.head()
test.head()
Pclass=pd.get_dummies(train['Pclass'],drop_first=True)

Pclass1=pd.get_dummies(test['Pclass'],drop_first=True)

Sex=pd.get_dummies(train['Sex'],drop_first=True)

Sex1=pd.get_dummies(test['Sex'],drop_first=True)

Embarked=pd.get_dummies(train['Embarked'],drop_first=True)

Embarked1=pd.get_dummies(test['Embarked'],drop_first=True)
train=pd.concat([train,Pclass,Sex,Embarked],axis=1)

test=pd.concat([test,Pclass1,Sex1,Embarked1],axis=1)
train.drop(['Sex','Embarked','Pclass'],axis=1,inplace=True)

test.drop(['Sex','Embarked','Pclass'],axis=1,inplace=True)
train.head()
test.head()
sample_sub.head()
y=train['Survived']
y.head()
X=train.drop('Survived',axis=1)
X.head()
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X = sc.fit_transform(X)

test = sc.fit_transform(test)
import keras

from keras.models import Sequential

from keras.layers import Dense
# Initialising the NN

model = Sequential()



# layers

model.add(Dense(9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))

model.add(Dense(9, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(5, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# summary

model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X, y, batch_size = 32, nb_epoch = 100)
y_pred = model.predict(test)

y_final = (y_pred > 0.5).astype(int).reshape(test.shape[0])
sample_sub['Survived']= y_final

sample_sub.to_csv("submit.csv", index=False)

sample_sub.head()