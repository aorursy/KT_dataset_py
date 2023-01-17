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
import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv('../input/heart.csv')
df.head()
df.isnull().sum()
df.describe()
df.target.value_counts()
plt.figure(figsize=(5,5))

sns.countplot(x='target',data=df)
pd.crosstab(df['age'],df['target']).plot(kind='bar',figsize=(10,5))

plt.title('For all age groups')

plt.xlabel('Age')

plt.ylabel('Frequency')
sns.countplot(x='sex',data=df,hue='target')
countFemale= len(df[df['sex']==0])

countMale = len(df[df['sex']==1])

l=len(df['sex'])

print('Percentage of male patients is {:.2f}'.format((countMale/l)*100))

print('Percentage of Female patients is {:.2f}'.format((countFemale/l)*100))
plt.figure(figsize=(10,5))

df['chol'].plot(kind='line',color='red',label='chol')

df['thalach'].plot(kind='line',color='blue',label='thalach',linestyle=':')

df['trestbps'].plot(kind='line',color='green',label='trestbps')

plt.legend()

plt.title('Heart Disease symptoms',fontsize=15)

plt.ylabel('data')

pd.crosstab(df['slope'],df['target']).plot(kind='bar',figsize=(10,5))

plt.title('Frequency according to slope')

plt.xlabel('The Slope of The Peak Exercise ST Segment ')



pd.crosstab(df['fbs'],df['target']).plot(kind='bar',figsize=(10,5))

plt.title('Frequency according to FBS')

plt.xlabel('Fasting Blood sugar (1:True,0:False)')

plt.figure(figsize=(12, 12))

sns.heatmap(df.corr(), annot=True, fmt='.1f')

cp =pd.get_dummies(df['cp'],prefix='cp')

thal=pd.get_dummies(df['thal'],prefix='thal')

slope = pd.get_dummies(df['slope'],prefix='slope')
train =pd.concat([df,cp,thal,slope],axis=1)

train=train.drop(['cp','thal','slope'],axis=1)
y=train['target']

X=train.drop(['target'],axis=1)
from sklearn.model_selection import train_test_split

from tpot import TPOTClassifier
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
tpot=TPOTClassifier(generations=10,population_size=100,verbosity=2)

tpot.fit(X_train,y_train)
tpot.score(X_test,y_test)