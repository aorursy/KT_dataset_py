# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

train_data=pd.read_csv('/kaggle/input/titanic/train.csv')

gender_submission_data=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train_data.columns
train_data.sample(3)
test_data.sample(3)
train_data.describe()
test_data.describe()
train_data.info()
train_data[train_data['Embarked'].isnull()]
train_data['Embarked']=train_data['Embarked'].fillna('S')
train_data['Age'].describe()
for name in train_data['Name']:

    train_data['Title']=train_data['Name'].str.extract('([A-Za-z]+)\.',expand=True)
df=train_data.groupby('Title').mean()



df.head()
train_data[train_data['Age'].isnull()].groupby('Title', as_index=False).count()
titles = ['Dr','Master','Miss','Mr','Mrs']



for title in titles:

    train_data.loc[train_data.Age.isnull() & (train_data.Title == title),'Age']=df['Age'][title]

train_data.isnull().sum()
for name in test_data['Name']:

    test_data['Title']=test_data['Name'].str.extract('([A-Za-z]+)\.', expand=True)
test_df=test_data.groupby('Title').mean()



title = ['Ms','Master','Miss','Mr','Mrs']



for title in test_data['Title']:

    test_data.loc[(test_data.Age.isnull()) & (test_data['Title'] == title),'Age']=test_df['Age'][title]

    

test_data.isnull().sum()
test_data.loc[test_data['Title']=='Ms']
test_data['Age']=test_data['Age'].fillna(21)



test_data.isnull().sum()
test_data[test_data.Fare.isnull()]
test_data.groupby('Pclass')['Fare'].median()
test_data['Fare']=test_data['Fare'].fillna(7.9)
test_data.isnull().sum()
train_data.groupby(['Sex', 'Survived'] )['Survived'].count().unstack('Sex').plot(kind='bar')     
train_data['Sex'] = train_data.Sex.apply(lambda x: 0 if x == "female" else 1)

test_data['Sex'] = test_data.Sex.apply(lambda x: 0 if x == "female" else 1)
train_data.drop('Cabin', axis=1)



test_data.drop ('Cabin', axis=1)
for age in test_data['Age']:

    

    test_data.loc[(test_data['Age'] < 18),'Is_child']=1

    test_data.loc[(test_data['Age'] >= 18),'Is_child']=0



test_data.loc[test_data['Is_child']==1].sample(5)
for age in train_data['Age']:

    

    train_data.loc[(train_data['Age']<18),'Is_child']=1

    train_data.loc[(train_data['Age']>= 18),'Is_child']=0



train_data.sample(5)
Keys = {'Capt': 1, 'Col': 1, 'Countess': 2, 'Don':1, 'Dr':1,'Dona':2, 'Jonkheer':1, 'Lady':2, 'Major':1, 'Mlle':3, 'Mme':2, 'Ms': 3, 'Rev': 1, 'Sir':1,'Mr':1,'Mrs':2,'Miss':3,'Master':4}



# Remap the values of the dataframe 

train_data= train_data.replace({'Title':Keys})

test_data=test_data.replace({'Title':Keys})
#Confirm if remapping worked

#test_data.head()
Emb_Keys={'C':1,'Q':2,'S':3}



train_data=train_data.replace({'Embarked':Emb_Keys})



test_data=test_data.replace({'Embarked':Emb_Keys})
train_data['Is_Alone']=train_data['SibSp']+train_data['Parch']



train_data['Is_Alone']=train_data.Is_Alone.apply(lambda x:1 if x == 0 else 0)



test_data['Is_Alone']=test_data['SibSp']+test_data['Parch']

test_data['Is_Alone']=test_data.Is_Alone.apply(lambda x:1 if x == 0 else 0)
train_data['Ind_Fare']=train_data['Fare']/(train_data['SibSp']+train_data['Parch']+1)

test_data['Ind_Fare']=test_data['Fare']/(test_data['SibSp']+test_data['Parch']+1)
train_data.sample(5)
train_data['Fam_Size']=train_data['SibSp']+train_data['Parch']

test_data['Fam_Size']=test_data['SibSp']+test_data['Parch']
plt.subplots(figsize = (15,10))

sns.heatmap(train_data.drop(columns='PassengerId').corr(), annot=True,cmap="RdYlGn_r")

plt.title("Feature Correlations", fontsize = 18)
from sklearn.ensemble import RandomForestClassifier



#Select features

y = train_data["Survived"]



features = ["Pclass", "Sex",'Fare','Title','Is_child','Is_Alone']

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=7)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")