import numpy as np

import pandas as pd

from subprocess import check_output

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics

import keras
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train.head(3)
df_test.head(3)
features_list = list(df_train.columns.values)

print(features_list)
#Use only important features

features = ['Pclass','Sex','Age','Fare']
#Processing data

le = LabelEncoder()



df_train["Sex"] =  le.fit_transform(df_train["Sex"])

df_test["Sex"] =  le.fit_transform(df_test["Sex"])



df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].mean())

df_test['Fare'] = df_train['Fare'].fillna(df_train['Fare'].mean())



df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

df_test['Age'] = df_train['Age'].fillna(df_train['Age'].mean())



df_train['Embarked'] = df_train['Embarked'].fillna("S")

df_test['Embarked'] = df_test['Embarked'].fillna("S")



df_train['Embarked'] = le.fit_transform(df_train['Embarked'])

df_test['Embarked'] = le.fit_transform(df_test['Embarked'])



df_train['Cabin'] = df_train['Cabin'].fillna("None")

df_test['Cabin'] = df_test['Cabin'].fillna("None")

df_train['Cabin'] = le.fit_transform(df_train['Cabin'])

df_test['Cabin'] = le.fit_transform(df_test['Cabin'])



df_train['Ticket'] = le.fit_transform(df_train['Ticket'])

df_test['Ticket'] = le.fit_transform(df_test['Ticket'])







y = df_train['Survived']

x = df_train[features]

x_t = df_test[features]
df_train.head(1)
df_test.head(1)
df_train.isnull().sum()
df_test.isnull().sum()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10,random_state=32)
print("X_train :",X_train.shape)

print("X_test :",X_test.shape)

print("y_train :",y_train.shape)

print("y_test :",y_test.shape)
df_train.head(1)
df_test.head(1)
from keras.models import Sequential

from keras.layers import Dense,Activation,Dropout

from keras.utils import np_utils

from keras.optimizers import SGD
model = Sequential()



model.add(Dense(64,input_dim=4))

model.add(Activation("relu"))

model.add(Dropout(0.3))



model.add(Dense(64))

model.add(Activation("relu"))

model.add(Dropout(0.3))



model.add(Dense(2))

model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
y_train_categorical = np_utils.to_categorical(y_train)
a = model.fit(X_train.values, y_train_categorical, nb_epoch=500)
y_test_categorical = np_utils.to_categorical(y_test)

loss_and_metrics = model.evaluate(X_test.values, y_test_categorical)

print(loss_and_metrics)
classes = model.predict_classes(x_t.values)
submission = pd.DataFrame({

    "PassengerId": df_test["PassengerId"],

    "Survived": classes})

print(submission[0:10])



submission.to_csv('./keras_model_3.csv', index=False)