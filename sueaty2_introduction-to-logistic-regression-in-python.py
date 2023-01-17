# Load libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()



# Load datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



dataset = [train, test]
train.shape, test.shape # One missing column in test would be 'Survived' feature.
train.head()
train.isnull().sum()

# ['Age'] is missing 177 values

# ['Cabin'] is missing 687 values

# ['Embarked'] is missing 2 values
test.isnull().sum()

# ['Age'] is missing 86 values

# ['Fare'] is missing 1 value

# ['Cabin'] is missing 327 values
def visualization(feature):

    O = train[train['Survived'] == 1][feature].value_counts()

    X = train[train['Survived'] == 0][feature].value_counts()

    visual_df = pd.DataFrame([O, X])

    visual_df.index = ['Survived','Dead']

    visual_df.plot(kind = 'bar',stacked = True, figsize = (12, 5), title = feature)
visualization('Pclass')

# Survived : Likely to have lived when Pclass == 1

# Dead : Likely to have died when Pclass == 3
visualization('Sex')

# Survived : Likely to have lived when female

# Dead : Likely to have died when male
visualization('Embarked')

# Embarked == S : More likely to have died

# Embarked == C : More likely to have survived

# Embarked == Q : More likely to have died
for vector in dataset :

    vector['FamilySize'] = vector['SibSp'] + vector['Parch'] + 1

train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean()
for vector in dataset :

    vector['Alone'] = 0 # Alone on board

    vector.loc[vector['FamilySize'] == 1, 'Alone'] = 1 # With Family

train[['Alone', 'Survived']].groupby(['Alone'], as_index = False).mean()
train = train.drop(['SibSp', 'Parch', 'FamilySize'], axis = 1)

test = test.drop(['SibSp', 'Parch', 'FamilySize'], axis = 1)
train['Embarked'].value_counts() + test['Embarked'].value_counts()
train['Embarked'] = train['Embarked'].fillna('S')

test['Embarked'] = test['Embarked'].fillna('S')
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace = True)
facet = sns.FacetGrid(train, aspect = 4)

facet.map(sns.kdeplot,'Fare', shade = True)

facet.set(xlim = (0, train['Fare'].max()))

facet.add_legend()

 

plt.show() 
train['Fare_Division'] = pd.qcut(train['Fare'], 4)

test['Fare_Division'] = pd.qcut(test['Fare'], 4)



train[['Fare_Division', 'Survived']].groupby(['Fare_Division'], as_index = False).mean()
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
train['Title'].value_counts() #  Mr,  Miss, Mrs are the majority
test['Title'].value_counts() # Mr, Miss, Mrs are the majority.
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)

test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
train['Age'].isnull().sum(), test['Age'].isnull().sum()
test[test['Age'].isnull()] # There's  no  other Ms. than her
train[train['Title'] == "Ms"]['Age'].value_counts() # Average age  of Ms in train data is 28
test.loc[test['Age'].isnull(), 'Age'] = 28
facet = sns.FacetGrid(train, aspect = 4)

facet.map(sns.kdeplot,'Age', shade = True)

facet.set(xlim = (0, train['Age'].max()))

facet.add_legend()

 

plt.show() 
train['Age_Division'] = pd.qcut(train['Age'], 5)

test['Age_Division'] = pd.qcut(test['Age'], 5)



train[['Age_Division', 'Survived']].groupby(['Age_Division'], as_index = False).mean()
train.head()
sexMap = {"male" : 0, "female" : 1}

train['Sex'] = train['Sex'].map(sexMap)

test['Sex'] = test['Sex'].map(sexMap)
embarkedMap = {"S" : 0, "C" : 1, "Q" : 2}

train['Embarked'] = train['Embarked'].map(embarkedMap)

test['Embarked'] = test['Embarked'].map(embarkedMap)
# Since  Mr, Miss, and Mrs were majority

titleMap = {"Mr" : 0, "Miss" : 1, "Mrs" : 2, "Master" : 3, "Dr" : 3, "Rev" : 3, "Mlle" : 3, "Col" : 3, "Major" : 3, 

                "Ms" : 3, "Lady" : 3, "Jonkheer" : 3, "Mme" : 3, "Capt" : 3, "Sir" : 3, "Don" : 3, "Countess" : 3}

train['Title'] = train['Title'].map(titleMap)

test['Title'] = test['Title'].map(titleMap)
# Fare

train.loc[train['Fare'] <= 7.91 , 'Fare'] = 0

train.loc[(7.91 < train['Fare']) & (train['Fare'] <= 14.454) , 'Fare'] = 1

train.loc[(14.454 < train['Fare']) & (train['Fare'] <= 31) , 'Fare'] = 2

train.loc[31 < train['Fare'] , 'Fare'] = 3

    

test.loc[test['Fare'] <= 7.91 , 'Fare'] = 0

test.loc[(7.91 < test['Fare']) & (test['Fare'] <= 14.454) , 'Fare'] = 1

test.loc[(14.454 < test['Fare']) & (test['Fare'] <= 31) , 'Fare'] = 2

test.loc[31 < test['Fare'] , 'Fare'] = 3
train.loc[train['Age'] <= 20 , 'Age'] = 0

train.loc[(20 < train['Age']) & (train['Age'] <= 26) , 'Age'] = 1

train.loc[(26 < train['Age']) & (train['Age'] <= 30) , 'Age'] = 2

train.loc[(30 < train['Age']) & (train['Age'] <= 38) , 'Age'] = 3

train.loc[38 < train['Age'] , 'Age'] = 4



test.loc[test['Age'] <= 20 , 'Age'] = 0

test.loc[(20 < test['Age']) & (test['Age'] <= 26) , 'Age'] = 1

test.loc[(26 < test['Age']) & (test['Age'] <= 30) , 'Age'] = 2

test.loc[(30 < test['Age']) & (test['Age'] <= 38) , 'Age'] = 3

test.loc[38 < test['Age'] , 'Age'] = 4
train = train.drop(['PassengerId', 'Ticket', 'Cabin', 'Fare_Division', 'Age_Division'], axis = 1)

test = test.drop(['Ticket', 'Cabin', 'Fare_Division', 'Age_Division'], axis = 1)
train.head()
train['tmp'] = 1

test['tmp'] = 1 

# 'tmp' are required to ease the computation later on.

# When computing weights for lost function, while other parameters are multiplied with each values of predictors,

# the first parameter is not. So, that's why we give a ['tmp'] = 1 so we  can still multiply from train/test_list but

# doesn't affect value.



train_df = pd.DataFrame(train, columns= ['tmp', 'Pclass', 'Sex', 'Age', 'Fare', 'Alone', 'Title'])

test_df = pd.DataFrame(test, columns= ['tmp', 'Pclass', 'Sex', 'Age', 'Fare', 'Alone', 'Title'])

target = train['Survived']



train_list = train_df.values.tolist()

train_list = np.array(train_list)



test_list = test_df.values.tolist()

test_list = np.array(test_list)



target_list = target.values.tolist()

target_list = np.array(target_list)
#Weight parameters are needed : w0, w1, w2... : w_list = []



import random

w_list = np.zeros(7)

for i in range(0, 7) :

    w_list[i] = random.random() * 2 - 1



#check randomly generated parameters

for i in range(0, 7) :

    print(w_list[i])



# set learning rate alpha : a

a = 0.0001
cnt = 0

# total iteration of 300,000

while(cnt < 300000) :

    Zi = train_list[:].dot(w_list) # Zi = w0 + w1 * x1 + w2 * x2 + ... + wn * xn   

    Hi = 1 / (1 + np.exp(-Zi)) # Hi = 1 / (1 + e^(-Zi)) 

    Hi_y = Hi - target_list



    for i in range(0, 7) :

        w_list[i] = w_list[i] - a * np.sum(train_list[:, i] * Hi_y) / 891

    cnt = cnt + 1
answer_list = []

for i in range (0, 418) :

    Zx = test_list[i].dot(w_list)

    if(Zx >= 0) :

        answer_list.append(1)

    else :

        answer_list.append(0)
submission_df = pd.DataFrame({

    "PassengerId" : test["PassengerId"],

    "Survived" : answer_list

})

submission_df.to_csv('submission.csv', index = False)