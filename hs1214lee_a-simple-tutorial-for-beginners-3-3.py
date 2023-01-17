import os

import pandas as pd



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

print('Shape of train dataset: {}'.format(train.shape))

print('Shape of test dataset: {}'.format(test.shape))
train['Type'] = 'train'

test['Type'] = 'test'

all = pd.concat([train, test], sort=False).reset_index(drop=True)

print('Shape of all dataset: {}'.format(all.shape))
print(all.isnull().values.any())

train.isnull().sum()
# Fill missing values

all_corr = all.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

all_corr
all.info()
all_corr[all_corr['level_0'] == 'Age']
all['Age'] = all.groupby(['Pclass', 'SibSp'])['Age'].apply(lambda x: x.fillna(x.median()))
print(all_corr[all_corr['level_0'] == 'Fare'])

all['Fare'] = all.groupby(['Pclass', 'Parch', 'SibSp'])['Fare'].apply(lambda x: x.fillna(x.median()))
all['Cabin'].value_counts()
all['Cabin'] = all['Cabin'].fillna('N')
print(all['Embarked'].value_counts())

all['Embarked'] = all['Embarked'].fillna('S')
# Check missing values again

all.isnull().values.any()
import numpy as np

import string



def extract_surname(data):  

    families = []

    

    for i in range(len(data)):        

        name = data.iloc[i]



        if '(' in name:

            name_no_bracket = name.split('(')[0] 

        else:

            name_no_bracket = name

            

        family = name_no_bracket.split(',')[0]

        title = name_no_bracket.split(',')[1].strip().split(' ')[0]

        

        for c in string.punctuation:

            family = family.replace(c, '').strip()

            

        families.append(family)

            

    return families



print(all['Name'].value_counts())

all['Title'] = all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

all['Family'] = extract_surname(all['Name'])

all['IsMarried'] = np.where(all['Title'] == 'Mrs', 1, 0)

all.drop(['Name'], inplace=True, axis=1)
all['Title']
all['Family']
all['FamilySize'] = all['SibSp'] + all['Parch'] + 1
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}

all['FamilySizeGrouped'] = all['FamilySize'].map(family_map)
all['Ticket_count'] = all.Ticket.apply(lambda x: all[all['Ticket']==x].shape[0])
all.info()
all.loc[(all['Sex'] == 'male') & (all['Age'] > 18.0), 'Sex'] = 0  # male

all.loc[(all['Sex'] == 'male') & (all['Age'] <= 18.0), 'Sex'] = 1 # boy

all.loc[all['Sex'] == 'female', 'Sex'] = 2                        # female
list1 = all[(all['Title'] != 'Mr') & (all['Survived'] == 0) ]['Ticket'].tolist() # Female and child no survide.

list2 = all[(all['Title'] == 'Mr') & (all['Survived'] == 1) ]['Ticket'].tolist() # Man survive.

all['Ticket_with_FC_dead'] = 0

all['Ticket_with_M_alive'] = 0

all.loc[all['Ticket'].isin(list1), 'Ticket_with_FC_dead'] = 1

all.loc[all['Ticket'].isin(list2), 'Ticket_with_M_alive'] = 1

all.drop(['Ticket'], inplace=True, axis=1)
all['Cabin'].value_counts()
all['Deck'] = all['Cabin'].apply(lambda x: x[0])

all['Deck'] = all['Deck'].replace(['T'], 'A')

all.drop(['Cabin'], inplace=True, axis=1)
all.loc[all['Deck'] == 'A', 'Deck'] = 0

all.loc[all['Deck'] == 'B', 'Deck'] = 1

all.loc[all['Deck'] == 'C', 'Deck'] = 2

all.loc[all['Deck'] == 'D', 'Deck'] = 3

all.loc[all['Deck'] == 'E', 'Deck'] = 4

all.loc[all['Deck'] == 'F', 'Deck'] = 5

all.loc[all['Deck'] == 'G', 'Deck'] = 6

all.loc[all['Deck'] == 'N', 'Deck'] = 7
all.loc[all['Embarked'] == 'S', 'Embarked'] = 0

all.loc[all['Embarked'] == 'Q', 'Embarked'] = 1

all.loc[all['Embarked'] == 'C', 'Embarked'] = 2
all['Title'].value_counts()
all['Title'] = all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')

all['Title'] = all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')

all['Title'].value_counts()
all.loc[all['Title'] == 'Mr', 'Title'] = 0

all.loc[all['Title'] == 'Miss/Mrs/Ms', 'Title'] = 1

all.loc[all['Title'] == 'Master', 'Title'] = 2

all.loc[all['Title'] == 'Dr/Military/Noble/Clergy', 'Title'] = 3
list1 = all[(all['Title'] != 0) & (all['Survived'] == 0) ]['Family'].tolist() # Female and child no survide.

list2 = all[(all['Title'] == 0) & (all['Survived'] == 1) ]['Family'].tolist() # Man survive.

all['Family_with_FC_dead'] = 0

all['Family_with_M_alive'] = 0

all.loc[all['Family'].isin(list1), 'Family_with_FC_dead'] = 1

all.loc[all['Family'].isin(list2), 'Family_with_M_alive'] = 1

all.drop(['Family'], inplace=True, axis=1)
all.loc[all['FamilySizeGrouped'] == 'Alone', 'FamilySizeGrouped'] = 0

all.loc[all['FamilySizeGrouped'] == 'Small', 'FamilySizeGrouped'] = 1

all.loc[all['FamilySizeGrouped'] == 'Medium', 'FamilySizeGrouped'] = 2

all.loc[all['FamilySizeGrouped'] == 'Large', 'FamilySizeGrouped'] = 3
train = all.loc[all['Type'] == 'train']

train.drop(['Type'], inplace=True, axis=1)

test = all.loc[all['Type'] == 'test']

test.drop(['Type', 'Survived'], inplace=True, axis=1)

print('Info of train dataset: {}'.format(train.info()))

print('Info of test dataset: {}'.format(test.info()))
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



train_y = train['Survived']

train_x = train.drop(['Survived', 'PassengerId'], axis=1)

model = LogisticRegression()

model.fit(train_x, train_y)

pred = model.predict(train_x)

metrics.accuracy_score(pred, train_y)
import time



test_x = test.drop('PassengerId', axis=1)

timestamp = int(round(time.time() * 1000))

pred = model.predict(test_x)

output = pd.DataFrame({"PassengerId":test.PassengerId , "Survived" : pred})

output = output.astype(int)

output.to_csv("submission_" + str(timestamp) + ".csv",index = False)