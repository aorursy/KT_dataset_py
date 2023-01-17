# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



#train_df.head()

#test_df.head()
train_df.info()

print("----------------------------------")

test_df.info()
train_df.drop(['PassengerId'], axis = 1, inplace = True) 

test_df.drop(['PassengerId'], axis = 1, inplace = True) 
train_df.head()

#test_df.head()


# Checking if any rows has all the null values.If yes then dropping the entire row.



#train_df.dropna(axis=0, how='all')

#test_df.dropna(axis=0, how='all')

#train_df.info()

#print("----------------------------------------")

#test_df.info()
train_df[train_df['Age'].isnull()]

train_df[train_df['Age'].isnull()].count()
test_df[test_df['Age'].isnull()]

test_df[test_df['Age'].isnull()].count()
train_df["Age"].mean()

#train_df["Age"].median()

#train_df['Age'].mode()
test_df['Age'].mean()

#test_df["Age"].median()

#test_df['Age'].mode()
train_df['Survived'].groupby(pd.qcut(train_df['Age'],6)).mean()
pd.qcut(train_df['Age'],6).value_counts()
train_df['Embarked'].unique()
train_df['Embarked'].value_counts()
sns.countplot(train_df['Embarked'])
train_df['Survived'].groupby(train_df['Embarked']).mean()
sns.countplot(train_df['Embarked'], hue=train_df['Pclass'])
train_df['Cabin_Letter'] = train_df['Cabin'].apply(lambda x: str(x)[0])
train_df['Cabin_Letter'].unique()
train_df['Cabin_Letter'].value_counts()
train_df['Survived'].groupby(train_df['Cabin_Letter']).mean()
train_df['Survived'].value_counts(normalize=True)
sns.countplot(train_df['Survived'],palette='Set2')
train_df['Pclass'].unique()
train_df['Survived'].groupby(train_df['Pclass']).count()
#train_df['Survived'].groupby(train_df['Pclass']).mean()
sns.countplot(train_df['Pclass'], hue=train_df['Survived'], palette= 'colorblind')
train_df['Sex'].value_counts(normalize=True)
#train_df['Survived'].groupby(train_df['Sex']).mean()
(train_df['Sex']).value_counts()
sns.countplot(train_df['Sex'],palette='cubehelix')
train_df['Survived'].groupby(train_df['Sex']).mean()
train_df['Name'].head()
train_df['Name_Head'] = train_df['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])

train_df['Name_Head'].value_counts()
train_df['Survived'].groupby(train_df['Name_Head']).mean()
plt.figure(figsize=(12,8))

sns.countplot(train_df['Name_Head'],palette='husl')
train_df['Survived'].groupby(train_df['Name_Head']).mean()
train_df['SibSp'].unique()
train_df['SibSp'].value_counts()
sns.countplot(train_df['SibSp'],palette='Set1')
train_df['Survived'].groupby(train_df['SibSp']).mean()
train_df['Parch'].unique()
train_df['Parch'].value_counts()
sns.countplot(train_df['SibSp'],palette='pastel')
train_df['Survived'].groupby(train_df['Parch']).mean()
test_df['Fare'].fillna(train_df['Fare'].mean(), inplace = True)
train_df['Fare'].unique()

train_df['Fare'].min()

train_df['Fare'].max()

#train_df['Fare'].mean()

#train_df['Fare'].mode()
pd.qcut(train_df['Fare'], 5).value_counts()
train_df['Survived'].groupby(pd.qcut(train_df['Fare'], 5)).mean()
pd.crosstab(pd.qcut(train_df['Fare'], 5), columns=train_df['Pclass'])
def names(train, test):

    for i in [train, test]:

        i['Name_Len'] = i['Name'].apply(lambda x: len(x))

        i['Name_Title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])

        del i['Name']

    return train, test
def age_impute(train, test):

    for i in [train, test]:

        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)

        data = train.groupby(['Name_Title', 'Pclass'])['Age']

        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))

    return train, test
def fam_size(train, test):

    for i in [train, test]:

        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',

                           np.where((i['SibSp']+i['Parch']) <= 3,'Nuclear', 'Big'))

        del i['SibSp']

        del i['Parch']

    return train, test
def cabin(train, test):

    for i in [train, test]:

        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])

        del i['Cabin']

    return train, test
def embarked_impute(train, test):

    for i in [train, test]:

        i['Embarked'] = i['Embarked'].fillna('S')

    return train, test
test_df['Fare'].fillna(test_df['Fare'].mean(), inplace = True)
def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked','Cabin_Letter' 'Name_Title', 'Fam_Size']):

    for column in columns:

        train[column] = train[column].apply(lambda x: str(x))

        test[column] = test[column].apply(lambda x: str(x))

        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]

        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)

        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)

        del train[column]

        del test[column]

    return train, test
def drop(train, test, bye = ['PassengerId']):

    for i in [train, test]:

        for z in bye:

            del i[z]

    return train, test
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train, test = names(train, test)

train, test = age_impute(train, test)

train, test = cabin(train, test)

train, test = embarked_impute(train, test)

train, test = fam_size(train, test)

test['Fare'].fillna(train['Fare'].mean(), inplace = True)

train, test = dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked','Cabin_Letter', 'Name_Title', 'Fam_Size'])
train.drop(['PassengerId'], axis = 1, inplace = True)
train.drop( ['Ticket'],axis=1,inplace = True)
train.info()
len(train.columns)
train.head()
test.info()
test.drop(['PassengerId'], axis = 1, inplace = True)
test.drop(['Ticket'], axis = 1, inplace = True)
len(test.columns)
test.info()
test.head()
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_features='auto',

                                oob_score=True,

                                random_state=1,

                                n_jobs=-1)
param_grid = { "criterion"   : ["gini", "entropy"],

             "min_samples_leaf" : [1, 5, 10],

             "min_samples_split" : [2, 4, 10, 12, 16],

             "n_estimators": [50, 100, 400, 700, 1000]}
gs = GridSearchCV(estimator=rf,

                  param_grid=param_grid,

                  scoring='accuracy',

                  cv=3,

                  n_jobs=-1

                 )
_gs = gs.fit(train.iloc[:, 1:], train.iloc[:, 0])
print(gs.best_score_)

print(gs.best_params_)
rf = RandomForestClassifier(criterion='gini', 

                             n_estimators=700,

                             min_samples_split=10,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1)

rf.fit(train.iloc[:, 1:], train.iloc[:, 0])

print ("%.4f" % rf.oob_score_ )
predictions = rf.predict(test)

predictions = pd.DataFrame(predictions, columns=['Survived'])
test = pd.read_csv(os.path.join('../input', 'test.csv'))

predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)

predictions.to_csv('y_test15.csv', sep=",", index = False)
# Still workinghhh