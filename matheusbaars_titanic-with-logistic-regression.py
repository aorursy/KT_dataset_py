# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head(5)
test.head(5)
sns.barplot(x='Pclass',y= 'Survived', data = train)
sns.barplot(x='Sex', y='Survived', data=train)
sns.countplot(x='Sex', data=train)
plt.figure(figsize=(50,10))

sns.barplot(x='Age',y='Survived', data=train)
train.isnull().sum()
test.isnull().sum()
train.info()
test.info()
#train.drop(['Cabin','Embarked','Name','Ticket'], inplace= True, axis=1)

#test.drop(['Cabin','Embarked','Name','Ticket'], inplace= True, axis=1)



train.drop(['Cabin','Name','Ticket'], inplace= True, axis=1)

test.drop(['Cabin','Name','Ticket'], inplace= True, axis=1)
train.isnull().sum()
age_class = train.groupby(['Sex','Pclass'])['Age'].mean()

age_class
def impute_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    Sex=cols[2]

    if pd.isnull(Age):

        if Pclass == 1 and Sex == 'male':

            return 41

        elif Pclass == 1 and Sex == 'female':

            return 34

        if Pclass == 2 and Sex == 'male':

            return 30

        elif Pclass == 2 and Sex == 'female':

            return 28

        if Pclass == 3 and Sex == 'male':

            return 26

        elif Pclass == 3 and Sex == 'female':

            return 21

    else:

        return Age

        
train['Age'] = train[['Age', 'Pclass', 'Sex']].apply(impute_age, axis = 1)
test['Age'] = test[['Age', 'Pclass', 'Sex']].apply(impute_age, axis = 1)
test['Fare'].mean()
test['Fare'].fillna(value=test['Fare'].mean(), inplace=True)
test.isnull().sum()
train.isnull().sum()
train = train.apply(lambda x:x.fillna(x.value_counts().index[0]))
train.head(2)
test.head(2)
#train=pd.get_dummies(train)

#test=pd.get_dummies(test)

#train.head()



train_dummy=pd.get_dummies(train['Embarked'],drop_first=True)

test_dummy=pd.get_dummies(test['Embarked'],drop_first=True)

#train_dummy.head()

train = pd.concat([train, train_dummy], axis=1)

test = pd.concat([test, test_dummy], axis = 1)

#test.head()

train.drop('Embarked', inplace=True, axis=1)

test.drop('Embarked', inplace=True, axis=1)

#test.head()

train_sex_dummy = pd.get_dummies(train['Sex'], drop_first=True)

test_sex_dummy = pd.get_dummies(test['Sex'], drop_first=True)

train = pd.concat([train, train_sex_dummy], axis=1)

test = pd.concat([test, test_sex_dummy], axis = 1)

train.drop('Sex', inplace=True, axis=1)

test.drop('Sex', inplace=True, axis=1)

test.head()
from sklearn.linear_model import LogisticRegression
X = train.drop(['PassengerId','Survived'], axis=1)

#X = train.drop('Survived', axis=1)

y = train['Survived']
lgmodel = LogisticRegression()

lgmodel.fit(X,y)
test.drop('PassengerId', axis=1, inplace=True)
lgmodel.predict(test)
sub=pd.DataFrame()

#sub['PassengerId']=test['PassengerId']

sub['PassengerId']=np.arange(892,1310)

sub['Survived']=lgmodel.predict(test)

sub.to_csv('submission-better-EmbarkedOn-DF.csv',index=False)