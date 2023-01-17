# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

gs=pd.read_csv("../input/gender_submission.csv")

y_train=train[['Survived']]

print(y_train.head())

x_train=train.drop('Survived', axis=1)

print(x_train.info())

print(test.info())
total=x_train.isnull().sum().sort_values(ascending=False)

percentage=round(total/len(x_train)*100,2)

pd.concat([total, percentage], axis = 1,keys= ['Total', 'Percentage'])
total=test.isnull().sum().sort_values(ascending=False)

percentage=round(total/len(test)*100,2)

pd.concat([total, percentage], axis = 1,keys= ['Total', 'Percentage'])
print(x_train.Embarked.value_counts(dropna=False))

x_train[x_train.Embarked.isnull()]
print(x_train['Fare'].where(x_train['Embarked']=='S').where(x_train['Pclass']==1).median())

print(x_train['Fare'].where(x_train['Embarked']=='C').where(x_train['Pclass']==1).median())

print(x_train['Fare'].where(x_train['Embarked']=='Q').where(x_train['Pclass']==1).median())
x_train.Embarked.fillna('C', inplace=True)

x_train.drop("Cabin",axis=1,inplace=True)

test.drop("Cabin",axis=1,inplace=True)

print(x_train.head())

print(test.head())
sns.boxplot(x="Pclass", y="Age", data=x_train)

plt.figure()

sns.scatterplot(x='Age',y='Fare',data=x_train,hue='Pclass',palette=['r','g','b'], legend="full")

plt.show()
total=[x_train,test]

for each in total:

    avr_age=each['Age'].mean()

    std_age=each['Age'].std()

    nan_age = each['Age'].isnull().sum()

    nan_age_random_list = np.random.randint(avr_age - std_age, avr_age + std_age, size=nan_age)

    each['Age'][np.isnan(each['Age'])] = nan_age_random_list

    each['Age'] = each['Age'].astype(int)
test['Fare'] = test['Fare'].fillna(test['Fare'].median())



x_train.drop("Name",axis=1,inplace=True)

test.drop("Name",axis=1,inplace=True)



x_train.drop("Ticket",axis=1,inplace=True)

test.drop("Ticket",axis=1,inplace=True)



codes1 = {'male':0, 'female':1}

x_train['Sex'] = x_train['Sex'].map(codes1)

test['Sex'] = test['Sex'].map(codes1)



codes2 = {'S':0, 'C':1, 'Q':2}

x_train['Embarked'] = x_train['Embarked'].map(codes2)

test['Embarked'] = test['Embarked'].map(codes2)
from keras import Sequential

from keras.layers import Dense



classifier = Sequential()

classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=len(x_train.columns)))

classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))

classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))



classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])



classifier.fit(x_train,y_train, batch_size=10, epochs=100)



eval_model=classifier.evaluate(x_train, y_train)

eval_model
y_pred=classifier.predict(test)

y_pred =np.array([round(x[0]) for x in y_pred])

y_pred=y_pred.astype(int)

print(y_pred)
ans = pd.DataFrame({'PassengerId' : test['PassengerId'] , 'Survived': y_pred})

ans.to_csv('submit.csv', index = False)

ans