# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

from itertools import chain
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import BatchNormalization

from keras.utils import np_utils

from keras.optimizers import Adam
train_data=pd.read_csv('../input/train.csv').drop(columns=['PassengerId','Name','Ticket','Cabin'])
train_data.head()
plt.hist(train_data.query('Fare>=0').Fare, bins=40)

plt.show()
sum(train_data['Fare']==0)
train_data.query('Fare==0')
train_data.groupby(["Embarked", "Pclass"]).Fare.mean()
train_data.loc[(train_data['Pclass'] == 1) & (train_data['Fare'] == 0.0),'Fare'] = 70.36

train_data.loc[(train_data['Pclass'] == 2) & (train_data['Fare'] == 0.0),'Fare'] = 20.33

train_data.loc[(train_data['Pclass'] == 3) & (train_data['Fare'] == 0.0),'Fare'] = 14.64
plt.hist(train_data.query('Age>=0').Age, bins=40)

plt.show()
train_data.Age.isna().sum()
means = train_data.groupby(['Sex', 'Pclass']).Age.mean()

train_data.Age = train_data.apply(lambda x: means[x.Sex][x.Pclass] if pd.isnull(x.Age) else x.Age, axis=1)
#one hot encode sex

train_data['Sex'] = pd.Categorical(train_data['Sex'])

dfDummies = pd.get_dummies(train_data['Sex'], prefix = 'category')

train_data = pd.concat([train_data.drop(columns=['Sex']), dfDummies], axis=1)

#one hot encode Pclass

train_data['Pclass'] = pd.Categorical(train_data['Pclass'])

dfDummies = pd.get_dummies(train_data['Pclass'], prefix = 'category')

train_data = pd.concat([train_data.drop(columns=['Pclass']), dfDummies], axis=1)

#one hot encode Embarked

train_data['Embarked'] = pd.Categorical(train_data['Embarked'])

dfDummies = pd.get_dummies(train_data['Embarked'], prefix = 'category')

train_data = pd.concat([train_data.drop(columns=['Embarked']), dfDummies], axis=1)
# Boarding Together or Alone

for i in range(len(train_data)):

    if train_data.loc[i, "SibSp"] + train_data.loc[i, "Parch"] == 0:

        train_data.loc[i, "Alone"] = 1

    else:

        train_data.loc[i, "Alone"] = 0



train_data.Alone = train_data.Alone.astype(int)
train_data.head()
features = ['Age','SibSp','Parch','Fare','category_female','category_male','category_1','category_2','category_3','category_C','category_Q','category_S','Alone']
y = train_data['Survived']

x = train_data.drop(columns=['Survived'])
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x = pd.DataFrame(x, columns=features)
x.head()
# create model

model = Sequential()

model.add(Dense(64, input_shape=(13,), activation='sigmoid'))

model.add(BatchNormalization())

model.add(Dropout(0.2))

model.add(Dense(64, activation='sigmoid'))

model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adadelta", loss='binary_crossentropy', metrics=["binary_accuracy"])

# Fit the model

model_result = model.fit(x, y, batch_size=100, epochs=200, validation_split= 0.2, shuffle = True)
print("<-------Final Metrics------->")

print("Loss     = ",model_result.history['val_loss'][199])

print("Accuracy = ",model_result.history['val_binary_accuracy'][199])
plt.figure(figsize=(20, 10))



plt.subplot(1, 2, 1)

plt.plot(model_result.history["loss"], label="training")

plt.plot(model_result.history["val_loss"], label="validation")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()



plt.subplot(1, 2, 2)

plt.plot(model_result.history["binary_accuracy"], label="training")

plt.plot(model_result.history["val_binary_accuracy"], label="validation")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()



plt.show()
test=pd.read_csv('../input/test.csv')

test_data=pd.read_csv('../input/test.csv').drop(columns=['PassengerId','Name','Ticket','Cabin'])
plt.hist(test_data.query('Fare>=0').Fare, bins=40)

plt.show()
sum(test_data['Fare']==0)
test_data.query('Fare==0')
test_data.groupby(["Embarked", "Pclass"]).Fare.mean()
test_data.loc[(test_data['Pclass'] == 1) & (test_data['Fare'] == 0.0),'Fare'] = 76.68

test_data.loc[(test_data['Pclass'] == 2) & (test_data['Fare'] == 0.0),'Fare'] = 23.06

test_data.loc[(test_data['Pclass'] == 3) & (test_data['Fare'] == 0.0),'Fare'] = 13.91
plt.hist(test_data.query('Age>=0').Age, bins=40)

plt.show()
test_data.Age.isna().sum()
means = test_data.groupby(['Sex', 'Pclass']).Age.mean()

test_data.Age = test_data.apply(lambda x: means[x.Sex][x.Pclass] if pd.isnull(x.Age) else x.Age, axis=1)
#one hot encode sex

test_data['Sex'] = pd.Categorical(test_data['Sex'])

dfDummies = pd.get_dummies(test_data['Sex'], prefix = 'category')

test_data = pd.concat([test_data.drop(columns=['Sex']), dfDummies], axis=1)

#one hot encode Pclass

test_data['Pclass'] = pd.Categorical(test_data['Pclass'])

dfDummies = pd.get_dummies(test_data['Pclass'], prefix = 'category')

test_data = pd.concat([test_data.drop(columns=['Pclass']), dfDummies], axis=1)

#one hot encode Embarked

test_data['Embarked'] = pd.Categorical(test_data['Embarked'])

dfDummies = pd.get_dummies(test_data['Embarked'], prefix = 'category')

test_data = pd.concat([test_data.drop(columns=['Embarked']), dfDummies], axis=1)
# Boarding Together or Alone

for i in range(len(test_data)):

    if test_data.loc[i, "SibSp"] + test_data.loc[i, "Parch"] == 0:

        test_data.loc[i, "Alone"] = 1

    else:

        test_data.loc[i, "Alone"] = 0



test_data.Alone = test_data.Alone.astype(int)
test_data.head()
scaler.fit(test_data)
test_data = scaler.transform(test_data)
test_data = pd.DataFrame(test_data, columns=features)
predict = model.predict_classes(test_data)
predict = list(chain.from_iterable(predict))
my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predict})

my_submission.to_csv('submission.csv', index=False)

print(my_submission)