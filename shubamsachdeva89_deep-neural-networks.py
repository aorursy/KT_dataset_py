# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd



import tensorflow as tf

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()

train.describe()
test.head()

train.describe()
train.isna().sum()

test.isna().sum()
corr = train.corr(method='pearson')

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='RdBu_r', annot=True, linewidth=0.2)
train['Family_size'] = train['SibSp'] + train['Parch']

test['Family_size'] = test['SibSp'] + test['Parch']
#train['Fare'] = train['Fare'].fillna(train['Fare'].mean())

test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
sns.distplot(train['Fare'], bins=40, kde=False)

plt.figure()

sns.distplot(test['Fare'], bins=40, kde=False)
#Fare is Left Skew

train['Fare'] = pd.qcut(train['Fare'], 4)

lbl_train = LabelEncoder()

train['Fare'] = lbl_train.fit_transform(train['Fare'])
test.isna().sum()
test['Fare'] = pd.qcut(test['Fare'], 4)

lbl_test = LabelEncoder()

test['Fare'] = lbl_test.fit_transform(test['Fare'])
test.head()
train['Embarked'].fillna('S', inplace=True)

train['Cabin'].fillna('N', inplace=True)

train['Age'] = train['Age'].fillna(train['Age'].mean())



test['Embarked'].fillna('S', inplace=True)

test['Cabin'].fillna('N', inplace=True)

test['Age'] = train['Age'].fillna(train['Age'].mean())
sns.distplot(train['Age'], bins=40, kde=False)

plt.figure()

sns.distplot(test['Age'], bins=40, kde=False)
train['Age'] = pd.qcut(train['Age'], 4)

lbl_train = LabelEncoder()

train['Age'] = lbl_train.fit_transform(train['Age'])
test['Age'] = pd.qcut(test['Age'], 4)

lbl_test = LabelEncoder()

test['Age'] = lbl_test.fit_transform(test['Age'])
sns.countplot(train['Age'])
train['Sex'] = train['Sex'].map({'female':0, 'male':1}).astype(int)

test['Sex'] = test['Sex'].map({'female':0, 'male':1}).astype(int)
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
train.head()
train['Cabin'] = train['Cabin'].apply(lambda x:x[0])
test['Cabin'] = test['Cabin'].apply(lambda x:x[0])
train.head()
lbl_train = LabelEncoder()

train['Cabin'] = lbl_train.fit_transform(train['Cabin'])
lbl_test = LabelEncoder()

test['Cabin'] = lbl_test.fit_transform(test['Cabin'])
test.head()
x_train = train.drop(['PassengerId', 'Survived', 'Name', 'SibSp', 'Parch', 'Ticket'], axis=1)

y_train = train['Survived']



x_test = test.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket'], axis=1)
x_train.head()
print(type(x_train))

x_train = np.asarray(x_train)

print(type(x_train))
y_train = np.asarray(y_train)

x_test = np.asarray(x_test)
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)
validation_size = 200



x_val = x_train[:validation_size]

x_train = x_train[validation_size:]



y_val = y_train[:validation_size]

y_train = y_train[validation_size:]
model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(32, activation='relu', input_shape=(7,)),

    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.Dense(32, activation='relu'),

    #tf.keras.layers.Dropout(0.10),

    tf.keras.layers.Dense(16, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val))
acc = history.history['acc']

val_acc = history.history['val_acc']



epochs = range(1, len(acc)+1)



plt.plot(epochs, acc, 'r')

plt.plot(epochs, val_acc, 'b')

plt.figure()

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc)+1)



plt.plot(epochs, loss, 'r')

plt.plot(epochs, val_loss, 'b')
predictions = model.predict_classes(x_test)

ids = test['PassengerId'].copy()

new_output = ids.to_frame()

new_output["Survived"]=predictions

new_output.head(20)
new_output.to_csv("my_submission.csv",index=False)