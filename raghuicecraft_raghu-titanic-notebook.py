"""

References or inspired by

1. https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python?scriptVersionId=1574825

2. https://www.kaggle.com/sinakhorami/titanic-best-working-classifier

3. https://www.kaggle.com/longyin2/titanic-machine-learning-from-disaster-0-842/notebook

"""

import pandas as pd

import numpy as np

import warnings

import matplotlib.pyplot as plt   

import seaborn as sns             # not used



from sklearn import preprocessing 

from sklearn.model_selection import GridSearchCV 

from sklearn.ensemble import RandomForestClassifier     # ensembling

from sklearn import cross_validation



%matplotlib inline

warnings.filterwarnings('ignore')
# train and test data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head(10)

test.head(10)

full_data = [train, test]

print(type(full_data))

print(type(train))

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    print(type(dataset['FamilySize']))

    break

## pre-processing

# check for missing values and update them accordingly



print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())



for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())



for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())



# Fill null values in Embarked

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())



# Fill null values in Dare

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())



# Fill null values in Age

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

train['CategoricalAge'] = pd.cut(train['Age'], 5)



print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())



## Clean the Data

# get the date to numerical type for Classification work



for dataset in full_data:

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']                               = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare']                                  = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    dataset.loc[ dataset['Age'] <= 16, 'Age']                          = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4



# select the features    

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)



test  = test.drop(drop_elements, axis = 1)



train = train.values

test  = test.values

# train the model

rf = RandomForestClassifier(criterion='gini', 

                             n_estimators=700,

                             min_samples_split=16,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1) 



rf.fit(train[:, 1:], train[:, 0])

#rf.fit(train.iloc[:, 1:], train.iloc[:, 0])

print("%.4f" % rf.oob_score_)
# csv submission

submit = pd.read_csv('../input/genderclassmodel.csv')

submit.set_index('PassengerId',inplace=True)



rf_res =  rf.predict(test)

submit['Survived'] = rf_res

submit['Survived'] = submit['Survived'].apply(int)

submit.to_csv('submit.csv')