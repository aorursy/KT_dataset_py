import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
x_test = pd.read_csv('../input/titanic/test.csv')

train = pd.read_csv('../input/titanic/train.csv')

y_test = pd.read_csv('../input/titanic/gender_submission.csv')
print("Dataframe train's Info")

train.info()

print('\n\n')

print("Dataframe x_test's Info")

x_test.info()

print('\n\n')

print("Dataframe y_test's Info")

y_test.info()
train.head()
sns.catplot(data=train[train['Survived'] == 1], x='Survived', hue='Sex', row='Embarked', col='Pclass', kind ='count', palette = 'coolwarm', hue_order=['male','female'])
sns.barplot(x='Sex', y='Survived', data=train)
sns.barplot(x='Pclass', y='Survived', data=train)
sns.barplot(x='Embarked', y='Survived', data=train)
train.isnull().sum()
train.isnull().sum()/len(train)
train = train.drop('Cabin', axis=1)
known_age = train[~train['Age'].isnull()]

known_age
known_age.corr()['Age'].sort_values()[:-1]
train.groupby('Pclass').mean()['Age']
train['Age'] = train.groupby('Pclass').transform(lambda group: group.fillna(group.mean()))['Age']
sns.heatmap(train.isnull(),vmin=0, vmax=1)
train = train.dropna()
sns.heatmap(train.isnull(),vmin=0, vmax=1)
g = sns.FacetGrid(data=train,col='Survived')

g.map(plt.hist,'Age')
train = train.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare'], axis = 1)
train.head()
sex = pd.get_dummies(train['Sex'])

embarked = pd.get_dummies(train['Embarked'])
train = pd.concat([train, sex, embarked],axis =1)

train = train.drop(['Sex', 'Embarked'], axis =1)
train.head()
x_test.isnull().sum()
x_test['Age'] = x_test.groupby('Pclass').transform(lambda group: group.fillna(group.mean()))['Age']
x_test = x_test.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare', 'Cabin'], axis = 1)
sex = pd.get_dummies(x_test['Sex'])

embarked = pd.get_dummies(x_test['Embarked'])
x_test = pd.concat([x_test, sex, embarked],axis =1)

x_test = x_test.drop(['Sex', 'Embarked'], axis =1)
x_test.head()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = train.drop('Survived',axis=1)

y_train = train['Survived'].values.reshape(-1,1)
scaler.fit(x_train)
x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)
x_train.shape
x_test.shape
y_test = y_test['Survived']
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation,Dropout

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
model = Sequential()

model.add(Dense(units=7,activation='relu'))

model.add(Dense(units=7,activation='relu'))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=x_train, 

          y=y_train, 

          epochs=200,

          validation_data=(x_test, y_test), verbose=1

          )
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
predictions = model.predict_classes(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
Submission = pd.read_csv('../input/titanic/gender_submission.csv')

Submission['Survived'] = predictions

Submission.to_csv('predictions.csv', index = False)