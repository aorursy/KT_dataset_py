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
import matplotlib.pyplot as pl

import seaborn as sns

from sklearn import tree

from sklearn.metrics import accuracy_score



%matplotlib inline



sns.set()
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic//test.csv')
train.head()
train.info()
train.isnull().sum()
train.describe()
pd.value_counts(train['Survived']).plot.bar()
sns.countplot(x='Survived', data = train);
test['Survived'] = 0

test[['PassengerId', 'Survived']].to_csv('no_survivors.csv', index = False)
sns.countplot(x='Sex', data = train);
sns.catplot(x='Survived', col='Sex', kind= 'count', data = train);
train.groupby(['Sex']).Survived.sum()
print("Survived female % is: ",train[train.Sex == 'female'].Survived.sum()/train[train.Sex == 'female'].Survived.count())

print("Survived male % is: ",train[train.Sex == 'male'].Survived.sum()/train[train.Sex == 'male'].Survived.count())
f,ax=pl.subplots(1,2,figsize=(16,7))

train['Survived'][train['Sex']=='male'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[0],shadow=True)

train['Survived'][train['Sex']=='female'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[1],shadow=True)

ax[0].set_title('Survived (male)')

ax[1].set_title('Survived (female)')



pl.show()
test['Survived'] = test.Sex == 'female'

test['Survived'] = test.Survived.apply(lambda x: int(x))

#test.head(n=20)
test[['PassengerId', 'Survived']].to_csv('women_survive.csv', index = False)
sns.catplot(x='Survived', col='Pclass', kind='count', data = train)
print("The percentage of survivals in") 

print("Pclass=1 : ", train.Survived[train.Pclass == 1].sum()/train[train.Pclass == 1].Survived.count())

print("Pclass=2 : ", train.Survived[train.Pclass == 2].sum()/train[train.Pclass == 2].Survived.count())

print("Pclass=3 : ", train.Survived[train.Pclass == 3].sum()/train[train.Pclass == 3].Survived.count())
sns.catplot('Pclass','Survived', kind='point', data=train);
pd.crosstab([train.Sex,train.Survived],train.Pclass,margins=True).style.background_gradient(cmap='summer_r')
sns.catplot('Pclass','Survived',hue='Sex', kind='point', data=train);
sns.catplot(x='Survived', col='Embarked', kind='count', data=train);
sns.catplot('Embarked','Survived', kind='point', data=train);
sns.catplot('Embarked','Survived', hue= 'Sex', kind='point', data=train);
sns.catplot('Embarked','Survived', col='Pclass', hue= 'Sex', kind='point', data=train);
pd.crosstab([train.Survived], [train.Sex, train.Pclass, train.Embarked], margins=True).style.background_gradient(cmap='autumn_r')
test['Survived'] = 0

# all women survived

test.loc[ (test.Sex == 'female'), 'Survived'] = 1

# except for those in Pclass 3 and embarked in S

test.loc[ (test.Sex == 'female') & (test.Pclass == 3) & (test.Embarked == 'S') , 'Survived'] = 0

test.head(n=20)
test[['PassengerId', 'Survived']].to_csv('embarked_pclass_sex.csv', index=False)
for df in [train, test]:

    df['Age_bin']=np.nan

    for i in range(8,0,-1):

        df.loc[ df['Age'] <= i*10, 'Age_bin'] = i
print(train[['Age' , 'Age_bin']].head(10))
sns.catplot('Age_bin','Survived',hue='Sex',kind='point',data=train);
sns.catplot('Age_bin','Survived', col='Pclass' , row = 'Sex', kind='point', data=train);
pd.crosstab([train.Sex, train.Survived], [train.Age_bin, train.Pclass], margins=True).style.background_gradient(cmap='autumn_r')