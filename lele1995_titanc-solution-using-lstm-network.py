import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



from keras.layers import Dense, LSTM, Activation, Dropout

from keras.models import Sequential

from keras.optimizers import Adam
train = pd.read_csv('../input/train.csv', index_col = ["PassengerId"])

test = pd.read_csv('../input/test.csv', index_col = ["PassengerId"])

combination = [train,test]



train.head()
train.describe(), test.describe()

train[["Survived","Pclass"]].groupby("Pclass").mean()



train[["Survived", "Sex"]].groupby("Sex").mean()
train['groups']=pd.cut(train.Age,[0,10,20,30,40,50,60,70,80])

train.head()
train[["Survived", "groups"]].groupby("groups").mean()
sns.barplot(train.groups, train.Survived)
sns.barplot(train.Sex, train.Survived)

sns.barplot(train.Pclass, train.Survived)
sns.barplot(train.Pclass, train.Survived, hue=train.Sex)
train.describe(include = ["O"]), test.describe(include = ["O"])
train=train.drop(["Name", "Ticket","Cabin", "groups"], axis=1)

test=test.drop(["Name", "Ticket","Cabin"], axis=1)

train.head()
male_female = {"male":1,

              "female":0}



train["Sex"]=train["Sex"].map(male_female)

test["Sex"]=test["Sex"].map(male_female)

train.head()
embar = {"C":2,

        "S":1,

        "Q": 0}

train["Embarked"]=train["Embarked"].map(embar)

test["Embarked"]=test["Embarked"].map(embar)

train.head()


train["Age"]=train["Age"].fillna(value=np.mean(train["Age"]))

test["Age"]=test["Age"].fillna(value=np.mean(train["Age"]))

test["Fare"]=test["Fare"].fillna(value=np.mean(train["Fare"]))

train["Embarked"]=train["Embarked"].fillna(value=round(np.mean(train["Embarked"])))

train_y = train["Survived"].iloc[:].values

train_x = train.drop(["Survived"], axis = 1).iloc[:,:].values



train_x = train_x.reshape(train_x.shape[0],-1,1)

train_x.shape
batch_size = 11

epoch = 20

hidden_units = 256 
model = Sequential()

model.add(LSTM(hidden_units, input_shape=train_x.shape[1:],batch_size=batch_size))

model.add(Activation('sigmoid'))

model.add(Dense(1))

model.compile(optimizer='Adam', loss = 'mean_squared_error',metrics = ['accuracy'] )

model.fit(train_x,train_y, batch_size=batch_size, epochs=epoch, verbose = 1)

out = pd.read_csv('../input/gender_submission.csv', index_col = ["PassengerId"])

y_test = out.iloc[:].values
test_x = test.iloc[:,:].values

test_x = test_x.reshape(test_x.shape[0],-1,1)

scores = model.evaluate(test_x, y_test, batch_size=batch_size)

predictions = model.predict(test_x, batch_size = batch_size)



print('LSTM test accuracy:', scores[1])