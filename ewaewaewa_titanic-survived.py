# Data processing

import pandas as pd



# Data visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Algorithm

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression
train = pd.read_csv('../input/train.csv')

test =  pd.read_csv('../input/test.csv')

all=[train,test]
train.head()
test.head()
train.shape #the train has 891 examples and 11 features + the target variable (survived)
test.shape
train.info() 
train.describe()
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train,palette='Blues')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train,palette='BuGn_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
sns.distplot(train['Age'],color='darkred',bins=30,kde=False);
sns.countplot(x='SibSp',data=train);
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False)
print('Train columns with null values:\n', train.isnull().sum())

print("*"*40)



print('Test/Validation columns with null values:\n', test.isnull().sum())

print("*"*20)
train = train.drop(['Cabin'],axis=1)

test=test.drop(['Cabin'],axis=1)

all=[train,test]
train.columns
train['Embarked'].unique()
print(train['Embarked'].value_counts())
train["Embarked"]=train['Embarked'].fillna('S')
train['Embarked'].isnull().any()
train['Embarked_cat'] = pd.factorize(train['Embarked'])[0]

test['Embarked_cat'] = pd.factorize(test['Embarked'])[0]

all=[train,test]
train = train.drop(['Embarked'],axis=1)

test=test.drop(['Embarked'],axis=1)

all=[train,test]
train.tail()
test.head()
# instaed pd.factorize

train.loc[train['Sex']=="male","Sex"]=0

train.loc[train['Sex']=="female","Sex"]=1

test.loc[test['Sex']=="male","Sex"]=0

test.loc[test['Sex']=="female","Sex"]=1



all=[train,test]
train.head()
test.head()
features = ["Age", "Survived","Fare", "Pclass", "SibSp", "Parch"]



print('MALES')



print(train[train.Sex == 0][features].describe())

print('_'*75)



print("FEMALES")

print(train[train.Sex == 1][features].describe())

print( train[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean() )
train['Age'].isnull().sum()
train['Age']=train['Age'].fillna(train['Age'].mean())

test['Age']=test['Age'].fillna(test['Age'].mean())

all=[train,test]
train['Age'].isnull().sum()
test['Age'].isnull().sum()
fig, ax = plt.subplots()



train['Age'].hist(bins=30,color='#A9C5D3', edgecolor='black',grid=False)



ax.set_title('Passanger\'s Age Histogram', fontsize=12)

ax.set_xlabel('Age', fontsize=12)

ax.set_ylabel('Frequency', fontsize=12)
train['Age_group'] = pd.cut(train['Age'], 5)

train[['Age_group', 'Survived']].groupby(['Age_group'],as_index=False).mean().sort_values(by='Age_group', ascending=True)
for dataset in all:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train.head(20)
test.head()
train = train.drop(['Age_group'], axis=1)

all = [train, test]

train.head()
for dataset in all:

    dataset['Family_size']=dataset['SibSp']+dataset['Parch']+1

train.head()
print(train[['Family_size','Survived']].groupby(['Family_size'],as_index=False).mean())
for dataset in all:

    dataset['Is_alone']=1

    dataset['Is_alone'].loc[dataset['Family_size']>1]=0

print(train[['Is_alone','Survived']].groupby(['Is_alone'],as_index=False).mean());
train.head()
feature_drop=['SibSp','Parch','Family_size']

train=train.drop(feature_drop,axis=1)

test=test.drop(feature_drop,axis=1)

all=[train,test]



train.head()
train = train.drop(['Name', 'PassengerId','Fare','Ticket'], axis=1)

test = test.drop(['Name','Fare','Ticket'], axis=1)

all = [train, test]

train.shape, test.shape
train.columns
test.columns
train.head()
test.head()
X_train=train.drop('Survived',axis=1)

Y_train=train['Survived']

X_test=test.drop('PassengerId',axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
log=LogisticRegression()

log.fit(X_train, Y_train)

acc_log=log.score(X_train,Y_train)



print('Logistic Regression accuracy is=' +str(round(acc_log,3)))