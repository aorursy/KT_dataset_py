# IMPORTS

import pandas as pd

import numpy





# INIT AND VARIABLES

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



print('Training on {} passengers'.format(len(train_data.values)))



train_data['Sex'] = train_data['Sex'].apply(lambda a: 1 if a=='female' else 0)

train_data['Age'] = train_data['Age'].apply(lambda a: 1 if a<=15 else 0)

train_data['Fare'] = train_data['Fare'].apply(lambda a: 1 if a>=numpy.mean(train_data['Fare'].values) else 0)



train_X = list(train_data[['Pclass',  'Age', 'Sex', 'SibSp', 'Parch', 'Fare']].values)

train_Y = list(train_data['Survived'].values)



# LEARN AND CREATE THE TREE

from sklearn import tree



clf = tree.DecisionTreeClassifier()

clf = clf.fit(train_X, train_Y)

print('Score of tree: {}'.format(clf.score(train_X, train_Y)))



print('Testing on {} passengers'.format(len(test_data.values)))

test_data['Sex'] = test_data['Sex'].apply(lambda a: 1 if a=='female' else 0)

test_data['Age'] = test_data['Age'].apply(lambda a: 1 if a<=15 else 0)

test_data['Fare'] = test_data['Fare'].apply(lambda a: 1 if a>=numpy.mean(train_data['Fare'].values) else 0)



test_X = list(test_data[['Pclass',  'Age', 'Sex', 'SibSp', 'Parch', 'Fare']].values)



results = clf.predict(test_X)
