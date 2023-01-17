# importing

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

%matplotlib inline
# read CSV file as DataFrame

train = pd.read_csv('../input/train.csv', index_col=0)

test = pd.read_csv('../input/test.csv', index_col=0)



# display the first 5 rows

train.head()
train.info()

print ('-------------------------------------')

test.info()
all_data = [train, test]



for dataset in all_data:

    # Create new feature FamllySize as a combination of SibSp and Parch

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # Create new feature IsAlone from FamilySize

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    # Remove all NULLS in Embarked, Fare, Age

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())

    # get average, std, and number of NaN values in Age column

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    # generate random numbers between (mean - std) & (mean + std)

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    # fill NaN values in Age column with random values generated

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    # convert from float to int

    dataset['Age'] = dataset['Age'].astype(int)

    # Mapping categorical data

    dataset['Sex'] = dataset['Sex'].map( {'female':0, 'male':1} ).astype(int)

    dataset['Embarked'] = dataset['Embarked'].map( {'S':0, 'C':1, 'Q':2} ).astype(int)
# drop unnecessary columns

drop_elements = ['Name', 'Ticket', 'Cabin']

train = train.drop(drop_elements, axis=1)

test = test.drop(drop_elements, axis=1)



train.info()
# define training and testing sets

X_train = train.drop('Survived', axis=1)

Y_train = train['Survived']

X_test = test.copy()



print (X_train.shape, Y_train.shape, X_test.shape)
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)