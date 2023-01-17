import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns





from sklearn.preprocessing import StandardScaler , MinMaxScaler , RobustScaler



from sklearn.model_selection import  train_test_split , cross_val_score







from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



from sklearn.svm import SVC



import os

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/train.csv' , index_col = 'PassengerId')

label = train['Survived']



test = pd.read_csv('../input/test.csv', index_col = 'PassengerId')

index = test.index
train.head(3)
train.info()
sns.countplot(label)
fig, ax =plt.subplots(1,3 , figsize=(10, 6) , sharex='col', sharey='row')

a = sns.countplot(x = 'Sex' , data=train , ax = ax[0] , order=['male' , 'female'])

b = sns.countplot(x = 'Sex' , data= train[label == 1] , ax = ax[1] , order=['male' , 'female'])

c = sns.countplot(x = 'Sex' , data= train[ ((train['Age'] < 21) & (label == 1)) ] , order=['male' , 'female'])

ax[0].set_title('All passenger')

ax[1].set_title('Survived passenger')

ax[2].set_title('Survived passenger under age 21')

fig, ax =plt.subplots(1,3 , figsize=(10, 6) , sharex='col', sharey='row')

a = sns.countplot(x = 'Pclass' , data=train , ax = ax[0] , order=[1 ,2,3])

b = sns.countplot(x = 'Pclass' , data= train[label == 1] , ax = ax[1] , order=[1 ,2,3])

c = sns.countplot(x = 'Pclass' , data= train[ ((train['Age'] < 21) & (label == 1)) ] , order=[1,2,3])

ax[0].set_title('All passanger')

ax[1].set_title('Survived passanger')

ax[2].set_title('Survived passanger under age 21')

fig, ax =plt.subplots(1,3 , figsize=(10, 6) , sharex='col', sharey='row')

a = sns.countplot(x = 'Embarked' , data=train , ax = ax[0] , order=['S' ,'Q','C'])

b = sns.countplot(x = 'Embarked' , data= train[label == 1] , ax = ax[1] , order=['S' ,'Q','C'])

c = sns.countplot(x = 'Embarked' , data= train[ ((train['Age'] < 21) & (label == 1)) ] , order=['S' ,'Q','C'])

ax[0].set_title('All passanger')

ax[1].set_title('Survived passanger')

ax[2].set_title('Survived passanger under age 21')
train['Deck'] = train.Cabin.str.get(0)

test['Deck'] = test.Cabin.str.get(0)

train['Deck'] = train['Deck'].fillna('NOTAVL')

test['Deck'] = test['Deck'].fillna('NOTAVL')

#Replacing T deck with closest deck G because there is only one instance of T

train.Deck.replace('T' , 'G' , inplace = True)

train.drop('Cabin' , axis = 1 , inplace =True)

test.drop('Cabin' , axis = 1 , inplace =True)
train.isna().sum()
test.isna().sum()
train.loc[train.Embarked.isna() , 'Embarked'] = 'S'
age_to_fill = train.groupby(['Pclass' , 'Sex' , 'Embarked'])[['Age']].median()

age_to_fill
for cl in range(1,4):

    for sex in ['male' , 'female']:

        for E in ['C' , 'Q' , 'S']:

            filll = pd.to_numeric(age_to_fill.xs(cl).xs(sex).xs(E).Age)

            train.loc[(train.Age.isna() & (train.Pclass == cl) & (train.Sex == sex) 

                    &(train.Embarked == E)) , 'Age'] =filll

            test.loc[(test.Age.isna() & (test.Pclass == cl) & (test.Sex == sex) 

                    &(test.Embarked == E)) , 'Age'] =filll
train.groupby(['Pclass' , 'Sex' , 'Embarked'])[['Age']].median()
train.Ticket = pd.to_numeric(train.Ticket.str.split().str[-1] , errors='coerce')

test.Ticket = pd.to_numeric(test.Ticket.str.split().str[-1] , errors='coerce')
Ticket_median = train.Ticket.median()

train.Ticket.fillna(Ticket_median , inplace =True)

test.Fare.fillna(train.Fare.median() , inplace =True)
train.isna().sum()
test.isna().sum()
train['Status'] = train['Name'].str.split(',').str.get(1).str.split('.').str.get(0).str.strip()

test['Status'] = test['Name'].str.split(',').str.get(1).str.split('.').str.get(0).str.strip()

importan_person = ['Dr' , 'Rev' , 'Col' , 'Major' , 'Mlle' , 'Don' , 'Sir' , 'Ms' , 'Capt' , 'Lady' , 'Mme' , 'the Countess' , 'Jonkheer' , 'Dona'] 

for person in importan_person:

    train.Status.replace(person, 'IMP' , inplace =True)

    test.Status.replace(person, 'IMP' , inplace =True)
train.Status.unique()
test.Status.unique()
train.head()
test.head()
test.drop(['Name' , 'Ticket' ] ,axis = 1, inplace = True)

train.drop(['Survived','Ticket' ,'Name' ], inplace =True , axis =1)
cat_col = ['Pclass' , 'Sex' , 'Embarked' , 'Status' , 'Deck']

train.Pclass.replace({

    1 :'A' , 2:'B' , 3:'C'

} , inplace =True)

test.Pclass.replace({

    1 :'A' , 2:'B' , 3:'C'

} , inplace =True)

train = pd.get_dummies(train , columns=cat_col)

test = pd.get_dummies(test , columns=cat_col)

print(train.shape , test.shape)
scaler = MinMaxScaler()



train= scaler.fit_transform(train)

test = scaler.transform(test)
model = RandomForestClassifier(bootstrap= True , min_samples_leaf= 3, n_estimators = 500 ,

                               min_samples_split = 10, max_features = "sqrt", max_depth= 6)

cross_val_score(model , train , label , cv=5)
model = LogisticRegression()

cross_val_score(model , train , label , cv=5)
from sklearn.svm import SVC

model = SVC(C=4)

cross_val_score(model , train , label , cv=5)
model.fit(train , label)

pre = model.predict(test)
ans = pd.DataFrame({'PassengerId' : index , 'Survived': pre})

ans.to_csv('submit.csv', index = False)

ans.head()