import numpy as np 

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

train.head()
print('Total number of passengers in training data..', len(train))

print('Number of passengers that survived..', len(train[train['Survived']==1]))
print('% of men who survived', 100*np.mean(train['Survived'][train['Sex']=='male']))

print('% of men who survived', 100*(1-np.mean(train['Survived'][train['Sex']=='male'])))

print('Number of men..', len(train[train['Sex']=='male']))
train['Sex'] = train['Sex'].apply(lambda x: 1 if x== 'male' else 0)

train['Age'] = train['Age'].fillna(np.mean(train['Age']))

train['Fare'] = train['Fare'].fillna(np.mean(train['Fare']))

train['Parch'] = np.where(train['Parch'] >= 1, 1, train['Parch'])

train['SibSp'] = np.where(train['SibSp'] >= 2, 2, train['SibSp'])

train.isnull().sum(axis = 0)

train['Elderly'] = np.where(train['Age']>=50, 1, 0)

train['Children'] = np.where(train['Age']<=10, 1, 0)

train['Adult'] = np.where((train['Age']<50) & (train['Age']>10), 1, 0)

train['Embarked'] = train['Embarked'].replace(['C'], 1)

train['Embarked'] = train['Embarked'].replace(['S'], 2)

train['Embarked'] = train['Embarked'].replace(['Q'], 3)

train['Embarked'] = train['Embarked'].fillna(2)

train['Fare'] = np.where(train['Fare'] >= 50, 50, train['Fare'])

train['Fare'] = np.where((train['Fare'] >= 10)&(train['Fare'] < 50), 10, train['Fare'])

train['Fare'] = np.where(train['Fare'] < 10, 0, train['Fare'])

train['Fare'] = train['Fare'].replace([50], 3)

train['Fare'] = train['Fare'].replace([10], 2)

train['Fare'] = train['Fare'].replace([0], 1)

train = train.drop(columns=['Ticket', 'Cabin', 'Name', 'Age'])

print(train.groupby('Parch')['PassengerId'].nunique())

print(train.groupby('SibSp')['PassengerId'].nunique())

print(train.groupby('Embarked')['PassengerId'].nunique())



train[:10]
import matplotlib.pyplot as plt

%matplotlib inline



labels = ["% Survived", "% Died"]

x = 100*np.mean(train['Survived'][train['SibSp']==2])

vals = [x,100-x]

plt.pie(vals, labels=labels, autopct='%0.0f%%')

plt.title("SibSp")

labels = ["% Survived", "% Died"]

vals = [100*np.mean(train['Survived'][train['Sex']==0]),100*(1-np.mean(train['Survived'][train['Sex']==0]))]



plt.pie(vals, labels = labels, autopct='%0.0f%%')

plt.axis("equal")

plt.title("Female")
labels = ["% Survived", "% Died"]

vals = [100*np.mean(train['Survived'][train['Sex']==1]),100*(1-np.mean(train['Survived'][train['Sex']==1]))]



plt.pie(vals, labels = labels, autopct='%0.0f%%')

plt.title("Male")
labels = ["% Survived", "% Died"]

x = 100*np.mean(train['Survived'][train['Embarked']==1])

vals = [x,100-x]

plt.pie(vals, labels=labels, autopct='%0.0f%%')

plt.title("Class C")
labels = ["% Survived", "% Died"]

x = 100*np.mean(train['Survived'][train['Embarked']==2])

vals = [x,100-x]

plt.pie(vals, labels=labels, autopct='%0.0f%%')

plt.title("Class S")
labels = ["% Survived", "% Died"]

x = 100*np.mean(train['Survived'][train['Embarked']==3])

vals = [x,100-x]

plt.pie(vals, labels=labels, autopct='%0.0f%%')

plt.title("Class Q")
labels = ["% Survived", "% Died"]

x = float(100*np.mean(train['Survived'][train['Parch']==0]))

vals = [x,100-x]

plt.pie(vals, labels=labels, autopct='%0.0f%%')

plt.title("Parch 0")
labels = ["% Survived", "% Died"]

x = float(100*np.mean(train['Survived'][train['Parch']==1]))

vals = [x,100-x]

plt.pie(vals, labels=labels, autopct='%0.0f%%')

plt.title("Parch 1")
labels = ["% Survived", "% Died"]

x = float(100*np.mean(train['Survived'][train['Elderly']==0]))

vals = [x,100-x]

plt.pie(vals, labels=labels, autopct='%0.0f%%')

plt.title("Elderly")
labels = ["% Survived", "% Died"]

x = float(100*np.mean(train['Survived'][train['Children']==0]))

vals = [x,100-x]

plt.pie(vals, labels=labels, autopct='%0.0f%%')

plt.title("Children")
labels = ["% Survived", "% Died"]

x = float(100*np.mean(train['Survived'][train['Adult']==0]))

vals = [x,100-x]

plt.pie(vals, labels=labels, autopct='%0.0f%%')

plt.title("Adult")
labels = ["% Survived", "% Died"]

x = 100*np.mean(train['Survived'][train['Fare']==1])

vals = [x,100-x]

plt.pie(vals, labels=labels, autopct='%0.0f%%')

plt.title("High Fare")
train = train.drop(columns=['PassengerId'])

train.head()

X = train.drop('Survived', axis=1)

y = train['Survived']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(max_depth = 3)

classifier.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

print('Training accuracy...', accuracy_score(y_train, classifier.predict(X_train)))

print('Testing accuracy...', accuracy_score(y_test, classifier.predict(X_test)))
test['Sex'] = test['Sex'].apply(lambda x: 1 if x== 'male' else 0)

test['Age'] = test['Age'].fillna(np.mean(test['Age']))

test['Fare'] = test['Fare'].fillna(np.mean(test['Fare']))

test['Parch'] = np.where(test['Parch'] >= 1, 1, test['Parch'])

test['SibSp'] = np.where(test['SibSp'] >= 2, 2, test['SibSp'])

test.isnull().sum(axis = 0)

test['Elderly'] = np.where(test['Age']>=50, 1, 0)

test['Children'] = np.where(test['Age']<=10, 1, 0)

test['Adult'] = np.where((test['Age']<50) & (test['Age']>10), 1, 0)

test['Embarked'] = test['Embarked'].replace(['C'], 1)

test['Embarked'] = test['Embarked'].replace(['S'], 2)

test['Embarked'] = test['Embarked'].replace(['Q'], 3)

test['Embarked'] = test['Embarked'].fillna(np.mean(test['Embarked']))

test['Fare'] = np.where(test['Fare'] >= 50, 50, test['Fare'])

test['Fare'] = np.where((test['Fare'] >= 10)&(test['Fare'] < 50), 10, test['Fare'])

test['Fare'] = np.where(test['Fare'] < 10, 0, test['Fare'])

test['Fare'] = test['Fare'].replace([50], 3)

test['Fare'] = test['Fare'].replace([10], 2)

test['Fare'] = test['Fare'].replace([0], 1)

test = test.drop(columns=['Ticket', 'Cabin', 'Name', 'Age'])
PassengerId = test['PassengerId']

test = test.drop(columns=['PassengerId'])
Predictions = classifier.predict(test)

Predictions
submission = pd.DataFrame({'PassengerId':PassengerId, 'Survived':Predictions})

filename = 'Titanic Prediction.csv'

submission.to_csv(filename, index=False)

print('Saved file:' + filename)
from keras.models import Sequential

from keras.layers import Dense , Dropout , Lambda, Flatten, BatchNormalization, Convolution2D , MaxPooling2D

from keras.optimizers import Adam ,RMSprop

from keras import  backend as K

from keras.layers.core import  Lambda , Dense, Flatten, Dropout



model = Sequential()



# layers

model.add(Dense(9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))

model.add(Dense(9, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(5, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# summary

model.summary()
# Compiling the NN

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Train the NN

model.fit(X_train, y_train, batch_size = 32, epochs = 200, verbose=0)
pred = model.predict(test)

pred = np.where(pred>=0.5, 1, pred)

pred = np.where(pred<0.5, 0, pred)

pred[0:15]
y_pred = model.predict(test)

y_final = (y_pred > 0.5).astype(int).reshape(test.shape[0])



output = pd.DataFrame({'PassengerId': PassengerId, 'Survived': y_final})

output.to_csv('Prediction Neural Network.csv', index=False)
output[0:15]