import pandas as pd

import seaborn as sns

import numpy as np

import random 

import re

import os

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.dummy import DummyClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

os.listdir('../input/titanic')
# Helper Function  - Code to get percent count for the missing data values

def missing_percent(data):

    Total = data.isnull().sum()

    Percent = Total/len(data)

    return pd.concat([Total,Percent],keys=['Total','Percent'],ignore_index=False,axis=1)
train_x = pd.read_csv('../input/titanic/train.csv')

test_x = pd.read_csv('../input/titanic/test.csv')
train_x.head()
train_x.info()
train_x.describe()
train_x.describe(include='O')
missing_percent(train_x)
pd.DataFrame(train_x.corr()['Survived'].sort_values(ascending=False))
fig = plt.figure(figsize=(12,4))

sns.barplot(y=train_x.corr()['Survived'],x=train_x.corr().index)
train_x['Sex'].value_counts()
sns.countplot(data=train_x,x='Sex')

plt.title('Count of Male and Female Passengers',fontsize=15)
sns.barplot(x='Survived',y='Age',data=train_x,hue='Sex')
train_x[['Sex','Survived']].groupby(['Sex']).mean().sort_values(by='Survived',ascending=False)
male_survived = train_x[train_x['Sex'] == 'male']['Survived'].tolist()

female_survived = train_x[train_x['Sex'] == 'female']['Survived'].tolist()
male_sample = random.sample(male_survived,50)

female_sample = random.sample(female_survived,50)

male_samplemean = np.mean(male_sample)

female_samplemean = np.mean(female_sample)

print ('Male Sample Mean : ', male_samplemean)

print ('Female Sample Mean : ', female_samplemean)

print ( 'Mean Differente : ', male_samplemean-female_samplemean)
import scipy.stats as stats

stats = stats.ttest_ind(male_sample,female_sample)

print ('The p Value is : ',format(stats.pvalue,'.32f'))
train_x[['Embarked','Survived']].groupby(['Embarked']).mean().sort_values(by='Survived',ascending=False)
train_x[['Embarked','Survived']].loc[train_x['Sex']=='female'].groupby(['Embarked']).mean().sort_values(by='Survived',ascending=False)
train_x[['Embarked','Survived']].loc[train_x['Sex']=='male'].groupby(['Embarked']).mean().sort_values(by='Survived',ascending=False)
fig = plt.figure(figsize=(17,4))

ax = sns.kdeplot(train_x.loc[train_x['Sex'] == 'male']['Age'].dropna(),color='green',shade=True,label='Male Age')

ax = sns.kdeplot(train_x.loc[train_x['Sex'] == 'female']['Age'].dropna(),color='gray',shade=True,label='Female Age')

plt.title('Age Distribution - Male V.S. Female', fontsize = 15)

plt.xlabel('Age')

plt.ylabel('test')

fig,ax = plt.subplots(figsize=(17,5))

sns.boxplot(x='Age',y='Sex',data=train_x)
fig = plt.figure(figsize=(12,5))

sns.violinplot(x='Embarked',y='Pclass',hue='Survived',data=train_x)
sns.catplot(x='Embarked',y='Pclass',hue='Survived',data=train_x,kind="violin", split=True)
sns.scatterplot(data=train_x,x='Age',y='PassengerId',hue='Survived')
sns.catplot(x='Pclass',y='Survived',data=train_x,kind='point')
sns.catplot(x='Survived',y='Fare',data=train_x,kind='point')
sns.countplot(train_x['Survived'],order=[1,0])
fig = plt.figure(figsize=(8,8))

sns.boxplot(y='Fare',x='Embarked',data=train_x)
fig = plt.figure(figsize=(8,8))

sns.boxplot(x='Embarked',y='Fare',hue='Pclass',data=train_x)
train_x[['Pclass','Survived']].groupby(['Pclass']).mean().sort_values(by='Survived',ascending=False)
train_x[['Survived','Embarked']].groupby(['Embarked']).mean().sort_values(by='Survived',ascending=False)
train_x[['Survived','SibSp']].groupby(['SibSp']).mean().sort_values(by='Survived',ascending=False)
train_x[['Survived','Parch']].groupby(['Parch']).mean().sort_values(by='Survived',ascending=False)
fig = plt.figure(figsize=(17,4))

sns.kdeplot(train_x[train_x['Pclass'] == 1]['Age'].dropna(),shade=True,color='green',label='First Class')

sns.kdeplot(train_x[train_x['Pclass'] == 2]['Age'].dropna(),shade=True,color='blue',label='Second Class')

sns.kdeplot(train_x[train_x['Pclass'] == 3]['Age'].dropna(),shade=True,color='red',label='Third Class')
train_x.head()
train_x.columns
train_x.shape
missing_percent(train_x)
missing_percent(test_x)
test_x.columns
test_x.shape
train_x.loc[train_x['Embarked'].isnull() == True]
train_x['Embarked'].value_counts()
train_x.loc[(train_x['Survived'] == 1)&(train_x['Pclass'] == 1 )&(train_x['SibSp'] == 0 )&(train_x['Parch'] == 0)&(train_x['Sex'] == 'female')]['Embarked'].value_counts()
train_x['Embarked'] = train_x['Embarked'].fillna('C')
test = train_x.loc[train_x['Age'].isnull() == True ]
test['Sex'].value_counts() 
test['Pclass'].value_counts()
test['Embarked'].value_counts()
test['Survived'].value_counts()
for val in test.itertuples():

    x = train_x.loc[(train_x['Survived'] == val.Survived)&(train_x['Pclass'] == val.Pclass )&(train_x['Embarked'] == val.Embarked)&(train_x['Sex'] == val.Sex)]['Age'].mean()

    train_x.loc[val.Index ,'Age']= x

#     test.loc[val.Index ,'Age']= x



train_x['Age'].isna().sum()
train_x['Cabin'] = train_x['Cabin'].fillna('N')

train_x['NewCabin'] = 'Allocated'

train_x.loc[train_x['Cabin'] == 'N','NewCabin'] = 'NotAlloted'

train_x[['NewCabin','Survived']].groupby(['NewCabin']).mean().sort_values(by='Survived',ascending=False)

sns.boxplot(data=train_x,x='Survived',y='Fare')
pd.qcut(train_x['Fare'],q=5).value_counts()
train_x.loc[train_x['Survived'] == 1,'Fare'].mean()
train_x['FareQuantile'] = pd.qcut(train_x['Fare'],q=5)
train_x[['FareQuantile','Survived']].groupby(['FareQuantile']).mean().sort_values(by='Survived',ascending=False)
sns.boxplot(data=train_x,x='Survived',y='Age')
train_x['AgeBand'] = pd.cut(train_x['Age'],bins=5)
train_x['NewFamily'] = train_x['SibSp']+train_x['Parch']

train_x.loc[train_x['NewFamily']>=1,'NewFamily'] = 'NotAlone'

train_x.loc[train_x['NewFamily']==0,'NewFamily'] = 'Alone'

train_x[['NewFamily','Survived']].groupby(['NewFamily']).mean().sort_values(by='Survived',ascending=False)

train_x['Title'] = train_x['Name'].apply(lambda x: re.findall('([A-Za-z]+)\.',x)[0])

title = pd.DataFrame(train_x['Title'].value_counts())

fig,ax = plt.subplots(figsize=(17,5))

sns.countplot(data=train_x,x='Title',ax=ax)
train_x['Title']= train_x['Title'].replace(['Dr', 'Rev', 'Mlle', 'Col', 'Major','Mme', 'Ms', 'Countess', 'Lady', 'Capt', 'Sir', 'Don', 'Jonkheer'],'rare')

train_x[['Title','Survived']].groupby(['Title']).mean()
Sex_code = {'male':1,'female':0}

train_x['Sex'] = train_x['Sex'].map(Sex_code)

train_x['AgeBand'] = pd.Categorical(train_x['AgeBand']).codes

train_x['Title'] = pd.Categorical(train_x['Title']).codes

train_x['FareQuantile'] = pd.Categorical(train_x['FareQuantile']).codes

train_x['Embarked'] = pd.Categorical(train_x['Embarked']).codes

train_x['NewCabin'] = pd.Categorical(train_x['NewCabin']).codes

train_x['NewFamily'] = pd.Categorical(train_x['NewFamily']).codes

train_x = train_x[['Pclass','Sex','NewFamily','Embarked', 'Title', 'AgeBand','FareQuantile','NewCabin','Survived']]

train_ip = train_x[['Pclass','Sex','NewFamily','Embarked', 'Title', 'AgeBand','FareQuantile','NewCabin']]

train_op = train_x['Survived']
X_train, X_test, y_train, y_test  = train_test_split(train_ip,train_op,test_size=0.3)

X_train.shape,y_train.shape
random = RandomForestClassifier()

random.fit(X_train,y_train)

score= round(random.score(X_test,y_test)*100,2)

print(score)
scores = []

Neighbours = []

for i in range (11):

    Neighbours.append(i+1)

    knn = KNeighborsClassifier(n_neighbors=i+1)

    knn.fit(X_train,y_train)

    knnscore = round((knn.score(X_test,y_test)*100),2)

    scores.append(knnscore)

Scores = pd.DataFrame({'Neigbbour':Neighbours,'Score':scores})

Scores
decesiontree = DecisionTreeClassifier()

decesiontree.fit(X_train,y_train)

decesiontreescore = decesiontree.score(X_test,y_test)

print(round(decesiontreescore*100,2))
adaboost = AdaBoostClassifier(learning_rate=0.2)

adaboost.fit(X_train,y_train)

ababoostscore = adaboost.score(X_test,y_test)

print(round(ababoostscore*100,2))
bagging = BaggingClassifier()

bagging.fit(X_train,y_train)

baggingscore = bagging.score(X_test,y_test)

print(round(baggingscore*100,2))
import h2o

from h2o.automl import H2OAutoML



h2o.init()
train_x =  h2o.import_file('../input/titanic/train.csv')

test_x =  h2o.import_file('../input/titanic/test.csv')
x = train_x.columns

y = 'Survived'

x.remove(y)
train_x[y] = train_x[y].asfactor()
aml = H2OAutoML(max_models=20, seed=1)

aml.train(x=x, y=y, training_frame=train_x)
lb = aml.leaderboard

lb.head(rows=lb.nrows)