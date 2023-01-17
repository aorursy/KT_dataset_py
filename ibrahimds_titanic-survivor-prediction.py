# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#make data into a dataframe
train_set = pd.read_csv('/kaggle/input/titanic/train.csv')
pd.set_option('display.max_columns', None)
train_set.head()
train_set.Cabin = train_set.Cabin.fillna('N0')
#creates a varible corresponding to the cabin letter of a passenger
train_set['Cabin_letter'] = train_set.Cabin.apply(lambda x: x[0])
train_set.head()
def find_number(Cabin):
    number = ''
    status = 'finding number'
    for char in Cabin:
        if char.isdigit():
                number += char
                status = 'reading number'
        else:
             if status == 'reading number':
                return int(number)
    if len(number) < 1:
        number = 0
    return int(number)
print(find_number('0C15'))
#plotting to determine if the cabin number matters for survival
train_set['Cabin_number'] = train_set.Cabin.apply(find_number)
train_set.pivot_table(index = 'Cabin_number', values= 'Survived').plot(kind = "bar")
#finds survial precentages for different Cabins letters
train_set.pivot_table(index = 'Cabin_letter', values = 'Survived')
#replaces NaN with average of age
train_set.Age = train_set.Age.fillna(np.nanmean(train_set.Age))
#converts sex to 1 if female and zero otherwise
train_set.Sex = (train_set.Sex == 'female')
train_set.Sex
#creates booleans for destinations
train_set['Cherbourg'] = (train_set.Embarked == 'C')
train_set['Queenstown'] = (train_set.Embarked == 'Q')
train_set.head()
#creates a new feature for each Cabin letter
for letter in train_set.Cabin_letter.unique():
    train_set['Cabin' + letter] = (train_set.Cabin_letter == letter)
#selects features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Parch', 'Cherbourg', 'Queenstown']
for letter in train_set.Cabin_letter.unique():
    features.append("Cabin" + letter)
X = train_set[features]
y = train_set.Survived
#spilts data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1)
from sklearn.tree import DecisionTreeClassifier
#creates and trains Decsion tree on Data
clf = DecisionTreeClassifier(min_samples_split = 10)
clf.fit(X_train,y_train)
#finds train and test accuracy
print(clf.score(X_test, y_test))
print(clf.score(X_train, y_train))