# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.columns
train.head(30)
train.describe()
train.info()
train.isnull().sum()
test.head()
C_sum = (train['Embarked']=='C').sum()

C_live = ((train['Embarked']=='C') & (train['Survived']==1)).sum()

C_live_rate = C_live / C_sum



S_sum = (train['Embarked']=='S').sum()

S_live = ((train['Embarked']=='S') & (train['Survived']==1)).sum()

S_live_rate = S_live / S_sum



Q_sum = (train['Embarked']=='Q').sum()

Q_live = ((train['Embarked']=='Q') & (train['Survived']==1)).sum()

Q_live_rate = Q_live / Q_sum
left = [1, 2, 3]

height = [C_live_rate, S_live_rate, Q_live_rate]

labels = ['Cherbourg', 'Queenstown', 'Southampton']



plt.bar(left, height, width=0.5, color='red',

        edgecolor='black', linewidth=2, tick_label=labels)

plt.show
P_1 = ((train['Pclass']==1) & (train['Survived']==1)).sum() / (train['Pclass']==1).sum()

P_2 = ((train['Pclass']==2) & (train['Survived']==1)).sum() / (train['Pclass']==2).sum()

P_3 = ((train['Pclass']==3) & (train['Survived']==1)).sum() / (train['Pclass']==3).sum()
left = [1, 2, 3]

height = [P_1, P_2, P_3]

labels = ['1st', '2nd', '3rd']



plt.bar(left, height, width = 0.5, color = 'red',

        edgecolor = 'black', linewidth = 2, tick_label = labels)
female = ((train['Sex']=='female') & (train['Survived']==1)).sum() / (train['Sex']=='female').sum()

male = ((train['Sex']=='male') & (train['Survived']==1)).sum() / (train['Sex']=='male').sum()
left = [1, 2]

height = [female, male]

labels = ['female', 'male']



plt.bar(left, height, width=0.5, color='red',

        edgecolor='black', linewidth=2, tick_label=labels)

plt.show()
import matplotlib.pyplot as plt



# generate data

x = np.random.rand(100)

y = np.random.rand(100)



fig = plt.figure()



ax = fig.add_subplot(1,1,1)



ax.scatter(x,y)



ax.set_title('first scatter plot')

ax.set_xlabel('x')

ax.set_ylabel('y')

x = train['Age']

y = train['Fare']

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.scatter(x,y)

ax.set_title('Fare & Age')

ax.set_xlabel('Age')

ax.set_ylabel('Fare')

plt.show()
P_1 = train['Age'][train['Pclass']==1].mean()

P_2 = train['Age'][train['Pclass']==2].mean()

P_3 = train['Age'][train['Pclass']==3].mean()

print(P_1)

print(P_2)

print(P_3)
train.loc[(train['Pclass']==1) & (train['Age'].isnull()), 'Age'] = P_1

train.loc[(train['Pclass']==2) & (train['Age'].isnull()), 'Age'] = P_2

train.loc[(train['Pclass']==3) & (train['Age'].isnull()), 'Age'] = P_3
train.isnull().sum()
S_n = (train['Embarked']=='S').sum()

C_n = (train['Embarked']=='C').sum()

Q_n = (train['Embarked']=='Q').sum()

print(S_n,C_n,Q_n)
train.loc[train['Embarked'].isnull(), 'Embarked']='S'
train.dropna(axis=1,inplace=True)

train = train.drop('Name',axis=1)

train = train.drop('Ticket',axis=1)
train.isnull().sum()
train = pd.get_dummies(train)
x = train.drop('Survived',axis=1)

x = x.drop('PassengerId',axis=1)

y = train['Survived']
from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=3)

clf = clf.fit(x,y)
predicted = clf.predict(x)

predicted
x
x_pre = test.drop('PassengerId',axis=1)

x_pre = x_pre.drop('Name',axis=1)

x_pre = x_pre.drop('Ticket',axis=1)

x_pre = x_pre.drop('Cabin',axis=1)



x_pre = pd.get_dummies(x_pre)

x_pre
x_pre.loc[(x_pre['Pclass']==1) & (x_pre['Age'].isnull()), 'Age'] = P_1

x_pre.loc[(x_pre['Pclass']==2) & (x_pre['Age'].isnull()), 'Age'] = P_2

x_pre.loc[(x_pre['Pclass']==3) & (x_pre['Age'].isnull()), 'Age'] = P_3

x_pre.loc[(x_pre['Fare'].isnull())] = x_pre['Fare'].mean()
x_pre.isnull().sum()
pred = clf.predict(x_pre)
pred = pd.DataFrame(pred)

pred
ID = pd.DataFrame(test['PassengerId'])
ID
submission = pd.concat([ID,pred],axis=1)

submission = submission.rename(columns={0: 'Survived'})

submission
submission.to_csv('submission.csv',index=False)