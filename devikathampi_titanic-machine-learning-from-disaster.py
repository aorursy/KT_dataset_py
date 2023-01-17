# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')

full_data =[train_data,test_data]

print(train_data.columns.values)
train_data[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',ascending=False)
train_data[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False)
train_data[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',ascending=False)
train_data[['Parch','Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived',ascending=False)
print("Before", train_data.shape, test_data.shape, full_data[0].shape, full_data[1].shape)
train_data=train_data.drop(['Ticket','Cabin'],axis=1)

test_data=test_data.drop(['Ticket','Cabin'],axis=1)

full_data=[train_data,test_data]
print("After", train_data.shape, test_data.shape, full_data[0].shape, full_data[1].shape)
for df in full_data:

    df['Title']=df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    

pd.crosstab(train_data['Title'],train_data['Sex'])
for df in full_data:

    df['Title']=df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title']=df['Title'].replace('Mlle','Miss')

    df['Title']=df['Title'].replace('Ms','Miss')

    df['Title']=df['Title'].replace('Mme','Miss')
train_data[['Title','Survived']].groupby(['Title'],as_index=False).mean()
title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}

for df in full_data:

    df['Title']=df['Title'].map(title_mapping)

    df['Title']=df['Title'].fillna(0)
train_data.sample(5)
train_data = train_data.drop(['Name','PassengerId'],axis=1)

test_data = test_data.drop(['Name'],axis=1)

full_data = [train_data, test_data]

train_data.shape, test_data.shape
for df in full_data:

    df['Sex']=df['Sex'].map({'female':1, 'male':0}).astype(int)

train_data.head()
test_data.head()
guess_ages=np.zeros((2,3))

guess_ages
for df in full_data:

    for i in range(0,2):

        for j in range(0,3):

            guess_df = df[(df['Sex']==i) & (df['Pclass']==j+1)]['Age'].dropna()

            

            age_guess = guess_df.median()

            

            guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5



    for i in range(0,2):

        for j in range(0,3):

            df.loc[ (df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1), 'Age'] = guess_ages[i,j]

    

    df['Age'] = df['Age'].astype(int)

    

train_data.head()
train_data['AgeBand'] = pd.cut(train_data['Age'],5)

train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending = True)
for df in full_data:

    df.loc[ df['Age'] <= 16, 'Age'] = 0 

    df.loc[ (df['Age']>16) & (df['Age']<= 32), 'Age']=1

    df.loc[ (df['Age']>32) & (df['Age']<= 48), 'Age']=2

    df.loc[ (df['Age']>48) & (df['Age']<= 64), 'Age']=3

    df.loc[ (df['Age']>64), 'Age']=4

train_data.head()
train_data = train_data.drop(['AgeBand'],axis=1)

full_data = [train_data, test_data]

train_data.head()
for df in full_data: 

    df['FamilySize'] = df["SibSp"] + df["Parch"] + 1

    

train_data[['FamilySize','Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=True)
for df in full_data:

    df['IsAlone'] = 0

    df.loc[df['FamilySize'] ==1 , 'IsAlone'] =1

    

train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_data= train_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_data = test_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

full_data = [train_data, test_data]



train_data.head()
for df in full_data:

    df['Age*Class']= df.Age * df.Pclass

    

train_data.loc[:,['Age*Class','Age','Pclass']].head(10)
freq_port = train_data.Embarked.dropna().mode()[0]

freq_port
for df in full_data:

    df['Embarked']=df['Embarked'].fillna(freq_port)

    

train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index= False).mean().sort_values(by ='Survived', ascending=False)
for df in full_data:

    df['Embarked'] = df['Embarked'].map( {'S':0, 'C':1, 'Q':2}).astype(int)

    

train_data.head()
test_data['Fare'].fillna(test_data['Fare'].dropna().median(), inplace=True)

test_data.head()
train_data['FareBand']=pd.qcut(train_data['Fare'],4)

train_data[['FareBand', 'Survived']].groupby(['FareBand'],as_index=False).mean().sort_values(by='FareBand', ascending=True)
for df in full_data:

    df.loc[ df['Fare'] <= 7.91, 'Fare']=0

    df.loc[ (df['Fare'] > 7.91) & (df['Fare'] <= 14.454) , 'Fare'] =1

    df.loc[ (df['Fare'] > 14.454) & (df['Fare'] <= 31.0) , 'Fare'] =2

    df.loc[ (df['Fare'] > 31.0), 'Fare'] =3

    df['Fare'] = df['Fare'].astype(int)

    

train_data = train_data.drop(['FareBand'], axis=1)

full_data = [train_data,test_data]

    

train_data.head(10)
test_data.head(10)
X_train = train_data.drop('Survived', axis=1)

Y_train = train_data['Survived']

X_test = test_data.drop('PassengerId', axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
from sklearn.ensemble import RandomForestClassifier



rand_forest = RandomForestClassifier()

rand_forest.fit(X_train, Y_train)

Y_pred = rand_forest.predict(X_test)

rand_forest.score(X_train, Y_train)

acc_rand = round(rand_forest.score(X_train, Y_train) * 100, 2)

acc_rand
output = pd.DataFrame({'PassengerId': test_data["PassengerId"], 'Survived': Y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")