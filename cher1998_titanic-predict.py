import numpy as np

import pandas as pd

import tensorflow

import featuretools as ft
from keras.models import Sequential

from keras.layers import Dense

# fix random seed for reproducibility

np.random.seed(7)
train_data=pd.read_csv('../input/train.csv')

test_data=pd.read_csv('../input/test.csv')
#train_data.describe()

train_data.isnull().sum()

train_data.describe()
train_data['Age']=train_data['Age'].fillna(train_data['Age'].mean())

train_data=train_data.drop(['Cabin'],axis=1)
train_data['Embarked']=train_data['Embarked'].fillna(method='ffill')
# female is 0 male 1

train_data['Sex']=train_data['Sex'].apply(lambda x:1 if x=='male' else 0 )

train_data['Embarked']=train_data['Embarked'].apply(lambda x:1 if x=='S'else 2 if x=='C' else 3)

train_data['Fare']=train_data['Fare'].apply(lambda x: x/513)

#train_data['Age']=train_data['Age'].apply(lambda x: x/79.6)
X=train_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

Y=train_data[['Survived']]
# create model

model = Sequential()

model.add(Dense(12, input_dim=7, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=75, batch_size=100)
# evaluate the model

scores = model.evaluate(X, Y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
test_data['Age']=test_data['Age'].fillna(test_data['Age'].mean())

test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].mean())

test_data['Fare']=test_data['Fare'].apply(lambda x: x/513)

#test_data['Age']=test_data['Age'].apply(lambda x: x/79.6)

test_data=test_data.drop(['Cabin'],axis=1)

test_data['Embarked']=test_data['Embarked'].fillna(method='ffill')

test_data['Sex']=test_data['Sex'].apply(lambda x:1 if x=='male' else 0 )

test_data['Embarked']=test_data['Embarked'].apply(lambda x:1 if x=='S'else 2 if x=='C' else 3)
test_data.isnull().sum()
X=test_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
predict=model.predict(X)
final_out=pd.DataFrame(columns=['PassengerId','Survived'])
final_out['PassengerId']=test_data['PassengerId']
final_out['Survived']=list(predict)
final_out['Survived']=final_out['Survived'].fillna(method='ffill')
final_out['Survived']=final_out['Survived'].apply(lambda x:int(round(x[0])))
final_out.reset_index()

final_out.to_csv('./final_predictions.csv',index=False)
final_out['Survived'].value_counts()