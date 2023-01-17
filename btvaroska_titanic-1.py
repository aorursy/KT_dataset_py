import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab as P
import csv as csv
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv('../input/train.csv', header=0)
def clean_data(d):
    'Clean data - use the same steps for train, cross-validation and test sets'
    r = d

    # Replace text field by number
    r['Gender']  = r['Sex'].map({'female':0, 'male':1}).astype(int)
    r = r.drop(['Sex'], axis = 1)

    #Replace Embarked by number
    r['Place'] = r['Embarked'].map({'S':0, 'C':1, 'Q':3})
    r.loc[(r.Embarked.isnull()), 'Place'] = 4
    r = r.drop(['Embarked'], axis = 1)

    # Fill blanks Age with median of whole group
    # TODO: split median by class and Gender
    r['AgeFill'] = r['Age']
    median_ages = np.zeros((2,3))
    for i in range (0,2):
       for j in range (0,3):
          median_ages[i,j] = r[(r['Gender'] == i) & (r['Pclass'] == j+1)]['Age'].dropna().median()
    for i in range (0,2):
       for j in range (0,3):
          r.loc[(r.Age.isnull()) & (r.Gender == i) & (r.Pclass == j+1), 'AgeFill'] = median_ages[i,j]
    r = r.drop(['Age'], axis = 1)

    # Size of family
    r['FamilySize'] = r['SibSp'] + r['Parch']
    r = r.drop(['SibSp', 'Parch'], axis = 1)

    # Drop all the rest for now
    # r = r.drop(['Name', 'Ticket', 'Fare', 'Cabin'], axis=1)

    return r
train = clean_data(train)
exception = train[(train['Survived'] == 0) & (train['Gender'] == 0)]
print(exception)
