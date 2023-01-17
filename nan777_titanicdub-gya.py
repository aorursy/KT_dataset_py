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
train = pd.read_csv('../input/train.csv')
train.head()

test = pd.read_csv('../input/test.csv')
test.head()

submission = pd.read_csv('../input/gender_submission.csv')
submission.head()

train['Cabin'].isnull().values.ravel().sum()

len(train)

train['Pclass'].isnull().values.ravel().sum()

train['Sex'].isnull().values.ravel().sum()

train['Age'].isnull().values.ravel().sum()

train['SibSp'].isnull().values.ravel().sum()

train['Parch'].isnull().values.ravel().sum()

train['Fare'].isnull().values.ravel().sum()

train['Embarked'].isnull().values.ravel().sum()
del train['PassengerId']
del train['Cabin']
del train['Ticket']
del train['Name']

train.head()

train = train.dropna(subset=['Embarked']) 

len(train)

age = train['Age']

age.head()

del train['Age']

train.head()

train = pd.get_dummies(train)

train.head()
local_train = np.asarray(train)
age = np.asarray(age)
age = np.reshape(age, (889,1))
local_train.shape

data = np.concatenate((local_train, age), axis = 1)

data_train = []
for i in range(0,889):
    if np.isnan(data[i][10]):
        data_train.append(data[i])

data_train = np.asarray(data_train)

df = pd.DataFrame(data, index = None)

df = df.dropna()

data = np.asarray(df)

x = data[:, 1:10]

y = data[:, 10:]



from sklearn.linear_model import LinearRegression

model1 = LinearRegression()
model1.fit(x, y)

x_test1 = data_train[:, 1:10]
x_test = data_train[:, :10]

y_test = model1.predict(x_test1)

y_test

y_test = np.round(y_test, decimals = 0)

for i in range(0,177):
    if y_test[i] <=0 :
        y_test[i] = abs(y_test[i])

data_new = np.concatenate((x_test, y_test), axis = 1)


data = np.concatenate((data, data_new), axis = 0)


y = data[:, 0:1]
x = data[:, 1:]


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)



from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train,y_train)

y_predict = model.predict(x_test)



from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_predict))



del test['Cabin']
del test['PassengerId']
del test['Name']
del test['Ticket']



test['Age'].isnull().values.ravel().sum()

test = pd.get_dummies(test)
age1 = test['Age']
age1 = np.asarray(age1)
del test['Age']
test1 = np.asarray(test)

age1 = np.reshape(age1, (418,1))


test1 = np.concatenate((test1, age1), axis = 1)

data_ = []
for i in range(0,418):
    if np.isnan(test1[i][9]):
        data_.append(test1[i])
