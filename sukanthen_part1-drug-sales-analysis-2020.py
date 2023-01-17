import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/hackerearth-effectiveness-of-std-drugs/dataset/train.csv')

test=pd.read_csv('/kaggle/input/hackerearth-effectiveness-of-std-drugs/dataset/test.csv')
train.head()
test.head()
train=train.drop(['patient_id'],axis=1)

test =test.drop(['patient_id'],axis=1)

print(train.shape,test.shape)
x = train['name_of_drug'].value_counts()

x
import seaborn as sns

sns.countplot(train['name_of_drug'].value_counts()>100)
sns.distplot(train['effectiveness_rating'])
a = train['effectiveness_rating'].value_counts()

print(a)

sns.countplot(train['effectiveness_rating'])
sns.jointplot(x='effectiveness_rating',y='number_of_times_prescribed',data=train)
train['name_of_drug'].value_counts().head(30).plot(kind='barh',figsize=(20,10))
import matplotlib.pyplot as plt

train['use_case_for_drug'].value_counts().head(40).plot(kind='barh',figsize=(10,10))
a =train['number_of_times_prescribed']

print('The mean value of number of prescribed drugs is:',a.mean())
print('The maximum number of prescribed drugs is:',a.max())

print('***********************************************')

print('The maximum number of prescribed drugs is:',a.min())
plt.plot(a)
x = train['name_of_drug'].value_counts().head(30)

y = train['number_of_times_prescribed'].head(30)

plt.plot(x,y)

plt.show()
corrmat = train.corr()

top=corrmat.index

plt.figure(figsize=(10,10))

graph = sns.heatmap(train[top].corr(),annot=True,cmap="Blues")