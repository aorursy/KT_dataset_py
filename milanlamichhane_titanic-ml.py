import numpy as np 

import pandas as pd 

import seaborn as sns

from matplotlib import pyplot as plt

from matplotlib import style



%matplotlib inline
from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



test = "/kaggle/input/test.csv"

test_data = pd.read_csv(test)

test_data.head(8)
train = "/kaggle/input/train.csv"

train_data = pd.read_csv(train)

train_data.head(8)
train_data.describe()
train_data.head(8)
survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = train_data[train_data['Sex']=='female']

men = train_data[train_data['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0])

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0])

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1])

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1])

ax.legend()

_ = ax.set_title('Male')
gender = {"male":1,"female":0}

data = [train_data, test_data]



for dataset in data:

    dataset['Sex_binary'] = dataset['Sex'].map(gender)

train_data.head(5)
ports = {"S": 0, "C": 1, "Q": 2}

data = [train_data, test_data]



for dataset in data:

    dataset['Embarked_new'] = dataset['Embarked'].map(ports)

train_data.head(5)
train_data=train_data.fillna(train_data.mean())

test_data=test_data.fillna(test_data.mean())

x_train= np.asarray(train_data[['Sex_binary','Fare',"Embarked_new"]])
y_train= np.asarray(train_data[['Survived']])
#x_test= np.asarray(test_data[['Sex_binary','Fare',"Embarked_new",'Survived']])

x_test= np.asarray(test_data[['Sex_binary','Fare',"Embarked_new"]])

logreg = LogisticRegression()

logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
accuracy = round(logreg.score(x_train, y_train) * 100, 2)

accuracy