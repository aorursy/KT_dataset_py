import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras
from keras.optimizers import SGD
import graphviz
#Loading the data

train_data = pd.read_csv('../input/titanic-machine-learning-from-disaster/train.csv')
test_data = pd.read_csv('../input/titanic-machine-learning-from-disaster/test.csv')

print(train_data.head())
print(test_data.head())
#Removing duplicates

train_data = train_data.drop_duplicates()
test_data = test_data.drop_duplicates()
#Checking for missing values in the data

train_data.isnull().sum()
test_data.isnull().sum()
#Filling missing values for features 'Fare' and 'Age' using imputation

train_data['Age'].fillna(train_data['Age'].mean(), inplace = True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace = True)

test_data['Fare'].fillna(test_data['Fare'].mean(), inplace = True)
#Creating a new feature "Relatives" by combining the features 'Sibsp' and 'Parch'

train_data['Relatives'] = train_data['SibSp'] + train_data['Parch']
test_data['Relatives'] = test_data['SibSp'] + test_data['Parch']
#Following my previous data analysis, I chose the following features to train the model.Since Pclass and Fare depict similar data regarding their socio-economic status I chose to keep Pclass.

input_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'Relatives']
X_train = train_data[input_features]
X_test = test_data[input_features]
Y_train = train_data['Survived']
#Converting categorical variables into dummy/indicator variables.

X_train = pd.get_dummies(X_train).astype(np.float64, copy=False)
X_test = pd.get_dummies(X_test).astype(np.float64, copy=False)
#Normalizing data 

from sklearn import preprocessing
scale = preprocessing.MinMaxScaler()

X_train = scale.fit_transform(X_train)
X_train = pd.DataFrame(X_train)

X_test = scale.fit_transform(X_test)
X_test = pd.DataFrame(X_test)

X_train.head(10)
model = keras.Sequential([
    keras.layers.Dense(16, activation= 'relu'),
	keras.layers.Dense(16, activation= 'relu'),
    keras.layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=50, batch_size=20)
model.summary()
Y_pred = model.predict(X_test)
Y_final = (Y_pred > 0.5).astype(int).reshape(X_test.shape[0])
print(Y_final)
model_1 = keras.Sequential([
    keras.layers.Dense(60,activation= 'relu'),(Dropout(0.5)),
    keras.layers.Dense(55,activation= 'relu'),(Dropout(0.5)),
    keras.layers.Dense(50,activation= 'relu'),(Dropout(0.5)),
	keras.layers.Dense(30,activation='relu'),(Dropout(0.5)),
    keras.layers.Dense(25,activation= 'relu'),( Dropout(0.5)),
    keras.layers.Dense(1, activation= 'sigmoid')
])

model_1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model_1.fit(X_train, Y_train, epochs=120, batch_size=32, validation_split=0.1)
model_1.summary()
Y_pred = model_1.predict(X_test)
Y_final = (Y_pred > 0.5).astype(int).reshape(X_test.shape[0])
print(Y_final)
model_2 = keras.Sequential([
    keras.layers.Dense(40,activation= 'relu'),(Dropout(0.4)),
    keras.layers.Dense(35,activation= 'relu'),(Dropout(0.4)),
    keras.layers.Dense(30,activation= 'relu'),(Dropout(0.4)),
    keras.layers.Dense(25,activation= 'relu'),(Dropout(0.4)),
    keras.layers.Dense(20,activation= 'relu'),(Dropout(0.4)),
	keras.layers.Dense(15,activation='relu'),(Dropout(0.4)),
    keras.layers.Dense(15,activation='relu'),(Dropout(0.4)),
    keras.layers.Dense(10,activation= 'relu'),( Dropout(0.4)),
    keras.layers.Dense(10,activation='relu'),(Dropout(0.4)),
    keras.layers.Dense(5,activation= 'relu'),(Dropout(0.4)),
    keras.layers.Dense(1, activation= 'sigmoid')
])

model_2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model_2.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2)
model_2.summary()
Y_pred = model_2.predict(X_test)
Y_final = (Y_pred > 0.5).astype(int).reshape(X_test.shape[0])
print(Y_final)
test_data=pd.read_csv('../input/titanic-machine-learning-from-disaster/test.csv')
print(len(test_data['PassengerId'].tolist()))
result_df=pd.DataFrame()
result_df['PassengerId']=test_data['PassengerId'].tolist()
result_df['Survived']= Y_final
print(result_df)

result_df.to_csv('/kaggle/working/submission.csv',index=False)
