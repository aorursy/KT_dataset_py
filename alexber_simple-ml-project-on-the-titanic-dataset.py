# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC



sns.set()

sns.set_palette("husl",2)

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

       

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv') #importing the data
train.info()
train.sample(10)
train.describe()
train.isna().sum()
data = train.copy() #We will work on a copy of our data

data.drop(['PassengerId','Cabin','Ticket'],axis=1,inplace=True)

data['Age'].fillna(data['Age'].median(),inplace=True)

data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)
data.isna().sum() #one more check
data['Family'] = data['SibSp'] + data['Parch'] #creating the family feature

data[['Family','SibSp','Parch']].sample(5) #quick check
#Engineering the Title feature by first parsing the names to extract titles

data['Title'] = data['Name'].apply(lambda x: x.partition(',')[2].partition('.')[0])

data[['Name','Title']].sample(5)#quick check
data['Title'].value_counts()#desplaying the number of occurences by values
rare_titles = (data['Title'].value_counts() < 10) #this line returns a Series with the titles as index and a column stating is the title is rare or not

rare_titles
#Replacing rare titles with 'Rare'

data['Title'] = data['Title'].apply(lambda x: 'Rare' if rare_titles[x] else x)

data['Title'].value_counts()
#now let's drop the column Name as we do not need it anymore

data.drop('Name',axis=1,inplace=True)
data.sample(5)# Quick look at our data
# Pclass

fig, axis = plt.subplots(1,2,figsize=(10,5))



sns.countplot(x='Pclass',data=data,ax = axis[0])

axis[0].set_title('Number of passenger per class')



sns.barplot(x='Pclass',y='Survived',data=data,ax = axis[1],ci=None)

axis[1].set_title('Survival rate per class')
#Gender feature

fig2, axis2 = plt.subplots(1,2,figsize=(10,5))



sns.countplot(x='Sex',data=data,ax = axis2[0])

axis2[0].set_title('Number of passenger per gender')



sns.barplot(x='Sex',y='Survived',data=data,ax = axis2[1],ci=None)

axis2[1].set_title('Survival rate per gender')
#Age feature

fig3, axis3 = plt.subplots(1,2,figsize=(10,5))



sns.boxplot(x='Age',data=data,ax = axis3[0])

axis3[0].set_title('Distribution of the Age feature')



sns.kdeplot(data[data.Survived==1].Age,shade=True,label='Alive',ax=axis3[1])

sns.kdeplot(data[data.Survived==0].Age,shade=True,label='Dead',ax=axis3[1])

axis3[1].set_title('Shape of the Age distribution')
#Family feature

fig4, axis4 = plt.subplots(1,2,figsize=(10,5))



sns.countplot(x='Family',data=data,ax = axis4[0])

axis4[0].set_title('Number of family per size category')



sns.barplot(x='Family',y='Survived',data=data,ax = axis4[1],ci=None)

axis4[1].set_title('Survival rate per family size')
#creation of the feature Alone

data['Alone'] = data['Family'].apply(lambda x : 0 if x > 0 else 1)
#examination of the Embarked feature

fig5, axis5 = plt.subplots(1,2,figsize=(10,5))



sns.countplot(x='Embarked',data=data,ax = axis5[0])

axis5[0].set_title('Boarding location ')



sns.barplot(x='Embarked',y='Survived',data=data,ax = axis5[1],ci=None)

axis5[1].set_title('Survival rate per boarding location')
fig5b, axis5b = plt.subplots(1,1,figsize=(10,5))

sns.violinplot(x = 'Embarked', y = 'Pclass', data=data,ax=axis5b)

axis5b.set_title('Pclass distribution per boarding location')
#Fare feature

fig6, axis6 = plt.subplots(1,2,figsize=(10,5))



sns.boxplot(x='Fare',data=data,ax = axis6[0])

axis6[0].set_title('Distribution of the Fare feature')



sns.kdeplot(data[data['Survived']==1]['Fare'],shade=True,label='Alive',ax=axis6[1])

sns.kdeplot(data[data['Survived']==0]['Fare'],shade=True,label='Dead',ax=axis6[1])

axis6[1].set_title('Shape of the Fare distribution')
fig6b, axis6b = plt.subplots(1,1,figsize=(5,10))

sns.boxplot(x='Pclass',y='Fare',data=data,ax=axis6b)

axis6b.set_title('Fare by Pclass')
#examination of the Title feature

fig7, axis7 = plt.subplots(1,2,figsize=(10,5))



sns.countplot(x='Title',data=data,ax = axis7[0])

axis7[0].set_title('Number of passenger per title category')



sns.barplot(x='Title',y='Survived',data=data,ax = axis7[1],ci=None)

axis7[1].set_title('Survival rate per title category')


def data_cleaning_and_feature_engineering(data):

    '''Preprocessing steps for the titanic dataset'''

    

    #filling NaN 

    data['Age'].fillna(data['Age'].median(),inplace=True)

    data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)

    

    #feature engineering

    data['Title'] = data['Name'].apply(lambda x: x.partition(',')[2].partition('.')[0])

    rare_titles = (data['Title'].value_counts() < 10) #this line returns a Series with the titles as index and a column stating is the title is rare or not

    data['Title'] = data['Title'].apply(lambda x: 'Rare' if rare_titles[x] else x)

    

    data['Family'] = data['SibSp'] + data['Parch'] #creating the family feature

    data['Alone'] = data['Family'].apply(lambda x : 0 if x > 0 else 1)





    #encode categoricals

    label = LabelEncoder()

    data['Title'] = label.fit_transform(data['Title'])

    data['Sex'] = label.fit_transform(data['Sex'])

    data['Embarked'] = label.fit_transform(data['Embarked'])

  

    #drop unused columns

    data.drop(['PassengerId','Cabin','Ticket','Name'],axis=1,inplace=True)



data2 = train.copy() #working on a fresh copy of the dataset

data_cleaning_and_feature_engineering(data2)#applying all preprocessing steps



X = data2.drop('Survived',axis=1,inplace=False)

y = data2['Survived']
#The selection of models is arbitrary

models =[] 

models.append(('XGBC',XGBClassifier()))

models.append(('RDF',RandomForestClassifier()))

models.append(('DT',DecisionTreeClassifier()))

models.append(('KN',KNeighborsClassifier()))

models.append(('SVM',SVC(gamma='auto')))
names = []

results = []

for name,model in models:

    kfold = StratifiedKFold(n_splits=5)

    cv_res = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

    results.append(cv_res)

    names.append(name)

    print(name, cv_res.mean(), cv_res.std())
param_test = {

"learning_rate": [0.1, 0.01],

'n_estimators' : [100, 500,1000],

'min_child_weight':[1, 3, 5, 7],

'max_depth': [2, 4, 7, 10],

'gamma': [0, 1,5],

'subsample':[0.3, 0.8, 1.0],

'colsample_bytree':[0.3, 0.8, 1.0]

}

search = GridSearchCV(estimator = XGBClassifier(), 

param_grid = param_test, scoring='accuracy',n_jobs=-1,iid=False, cv=5)



search.fit(X,y)

print(search.best_params_, search.best_score_)
param_test2 = {

"learning_rate": [0.1,0.06],

'n_estimators' : [900, 1000, 1100],

'min_child_weight':[4,5,6],

'max_depth': [6,7,8],

'gamma': [0.9,1],

'subsample':[0.7,0.8,0.9],

'colsample_bytree':[0.7,0.8,0.9]

}

search = GridSearchCV(estimator = XGBClassifier(), 

param_grid = param_test2, scoring='accuracy',n_jobs=-1,iid=False, cv=5)



search.fit(X,y)

print(search.best_params_, search.best_score_)
param_test3 = {

"learning_rate": [0.1],

'n_estimators' : [1000],

'min_child_weight':[5],

'max_depth': [7],

'gamma': [0,4,0.7,0.8, 0.9],

'subsample':[0.9],

'colsample_bytree':[0.9]

}

search = GridSearchCV(estimator = XGBClassifier(), 

param_grid = param_test3, scoring='accuracy',n_jobs=-1,iid=False, cv=5)



search.fit(X,y)

print(search.best_params_, search.best_score_)