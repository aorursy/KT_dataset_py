# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style = 'whitegrid')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')

data_train.head()
data_train.info()

print('----------------------------')

data_test.info()
# drop unnecessary columns, these columns won't be useful in analysis and prediction

#data_train = data_train.drop(['PassengerId'])

#data_train
fig , ax = plt.subplots(3,3,figsize=(15,15))

sns.countplot(x = 'Pclass', data = data_train,ax = ax[0,0])

sns.countplot(x = 'Embarked', data = data_train,ax = ax[0,1])

sns.countplot(x = 'Sex', data = data_train,ax = ax[0,2])

sns.countplot(x = 'Sex', hue = 'Survived',data = data_train,ax = ax[1,0])

sns.countplot(x = 'Embarked', hue = 'Survived',data = data_train,ax = ax[1,1])

sns.countplot(x = 'Embarked', hue = 'Pclass',data = data_train,ax = ax[1,2])

embark_perc = data_train[['Embarked', 'Survived']].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=ax[2,0])

pclass_perc = data_train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean()

sns.barplot(x='Pclass', y='Survived', data=pclass_perc,order=[1,2,3],ax=ax[2,1])

sex_perc = data_train[['Sex', 'Survived']].groupby(['Sex'],as_index=False).mean()

sns.barplot(x='Sex', y='Survived', data=sex_perc,order=['male','female'],ax=ax[2,2])

sns.factorplot(x='Embarked',y='Survived',data=data_train,size = 4, aspect=3)
#embarked

data_train.Embarked = data_train.Embarked.fillna('S')

data_test.Embarked = data_test.Embarked.fillna('S')

dummies_Embarked = pd.get_dummies(data_train.Embarked)

#dummies_Embarked.drop('S',axis = 1,inplace = False)

dummies_Embarked
plt.figure(1)

plt.subplot2grid((2,3),(0,0))

data_train.Survived.value_counts().plot(kind = 'bar')

plt.title('Survived or Dead')

plt.ylabel('num')



plt.subplot2grid((2,3),(0,1))

data_train.Pclass.value_counts().plot(kind = 'pie')

plt.title('Ticket class')