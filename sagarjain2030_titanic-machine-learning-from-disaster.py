# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import keras
train_data = pd.read_csv("../input/train.csv")
train_data.head()
train_data['SibSp'].fillna(0)
train_data['Parch'].fillna(0)
train_data['family'] = train_data['SibSp'] + train_data['Parch']
train_data.head()
from copy import deepcopy
overallData = deepcopy(train_data)
label_data = overallData['Survived']
feature_data = overallData[['Pclass','Sex','family','Embarked']]
label_data.shape
print()
feature_data.shape
X_train = feature_data[100:]
X_test = feature_data[:100]
y_train = label_data[100:]
y_test = label_data[:100]
y_train
df = pd.get_dummies(X_train['Pclass'],'Pclass')
X_training = deepcopy(X_train)
X_training = X_training.join(df)
X_training.drop('Pclass',1,inplace=True)
X_training = pd.get_dummies(X_training)
X_training
df = pd.get_dummies(X_test['Pclass'],'Pclass')
X_testing = deepcopy(X_test)
X_testing = X_testing.join(df)
X_testing.drop('Pclass',1,inplace=True)
X_testing = pd.get_dummies(X_testing)
X_testing
y_training = pd.get_dummies(y_train)
y_training
y_testing = pd.get_dummies(y_test)
y_testing
X_training.shape
from keras.models import Sequential
from keras.layers import Dense, Dropout

# define the model
model = Sequential()
model.add(Dense(512,input_shape= (9,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

# summarize the model
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy'])
import numpy as np
xTrain = np.array(X_training)
yTrain = np.array(y_training)
xTest = np.array(X_testing)
yTest = np.array(y_testing)
print(xTrain.shape)
print(yTrain.shape)
print(xTest.shape)
print(yTest.shape)
score = model.evaluate(xTest, yTest, verbose=1)
score
model.fit(xTrain, yTrain, batch_size=64, epochs=150,verbose=0)
score = model.evaluate(xTrain, yTrain)
print("\n Training Accuracy:", score[1])
score = model.evaluate(xTest, yTest)
print("\n Testing Accuracy:", score[1])
print(score)
test_data = pd.read_csv("../input/test.csv")
test_data.head()
test_data['SibSp'].fillna(0)
test_data['Parch'].fillna(0)
test_data['family'] = test_data['SibSp'] + test_data['Parch']
test_data.head()
testingDataGiven = deepcopy(test_data)
testing_data = testingDataGiven[['Pclass','Sex','family','Embarked']]
df = pd.get_dummies(testing_data['Pclass'],'Pclass')
realTestDataX = deepcopy(testing_data)
realTestDataX = realTestDataX.join(df)
realTestDataX.drop('Pclass',1,inplace=True)
realTestDataX = pd.get_dummies(realTestDataX)
realTestDataX
xTestData = np.array(realTestDataX)
SurviverOutput = model.predict(xTestData)
SurviverOutput
S = SurviverOutput[:,1:]
pred = [1 if s > 0.45 else 0 for s in S]
pred
newDF = pd.DataFrame()
newDF['PassengerId'] = test_data['PassengerId']
newDF['Survived'] = pred
print(newDF.to_csv('TitanicWithNeuralNetwork.csv',index=False))
