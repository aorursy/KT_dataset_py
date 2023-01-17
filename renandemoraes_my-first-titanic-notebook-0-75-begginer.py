import numpy as np 

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()
train.head()
train.info()
test.info()
# Dropping Variables I don't wanna conisder to run the algorithm

train.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)

test.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)

train.head()
test.head()
#Checking outliers

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.boxplot(data=train, x='Age', showmeans=True )

plt.subplot(2,2,2)

sns.boxplot(data=test, x='Age', showmeans=True)
#dropping some of them

rows = train[train['Age']>65].index

train.drop(rows, inplace=True)
rows1 = test[test['Age']>65].index

test.drop(rows1, inplace=True)
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.boxplot(data=train, x='Age', showmeans=True )

plt.subplot(2,2,2)

sns.boxplot(data=test, x='Age', showmeans=True)
#Exploring NaN values

train.isnull().sum()
test.isnull().sum()
#Treating NaN Values on Train "Embarked"

train1 = train.dropna(axis=0, subset=['Embarked'])

test1 = test.dropna(axis=0, subset=['Fare'])
train1.isnull().sum()
test1.isnull().sum()
#Now filling the NaN values on variable "Age" on train1 and test1

train1.describe()
test1.describe()
train1['Age'].fillna(30, inplace=True)

test['Age'].fillna(30, inplace=True)
#checking NaN values

test.isnull().sum()
train1.isnull().sum()
#Transform cat to num values - 'Sex' and 'Embarked'

train1.info()
test.info()
#Transforming Variables sex to bin on df's train1 and test1

def sex_to_bin(valor):

    if valor == 'female':

        return 0

    else:

        return 1



train1['sex_to_bin'] = train1['Sex'].map(sex_to_bin)
train1.drop(['Sex'], axis=1, inplace=True)

train1.info()
def sex_to_bin(valor):

    if valor == 'female':

        return 0

    else:

        return 1



test['sex_to_bin'] = test['Sex'].map(sex_to_bin)
test.drop(['Sex'], axis=1, inplace=True)

test.head()
#Transforming cat to num in "Embarked"

port = {"S": 0, "C": 1, "Q": 2}

data = [train1]

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(port)
train1.info()
port = {"S": 0, "C": 1, "Q": 2}

data = [test]

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(port)
test.info()
#Separating the variables and applying the Random Forest algorythm

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
variables = ["sex_to_bin", "Age", "Embarked", 'Pclass']
x = train1[variables]

y = train1["Survived"]
y.head()

#x.head()
model.fit(x,y)
x_prev = test[variables]

#x_prev.head()

x_prev.info()
p = model.predict(x_prev)

p