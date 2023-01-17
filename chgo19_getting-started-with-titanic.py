# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('../input/titanic/train.csv')

train_data.head()
test_data = pd.read_csv('../input/titanic/test.csv')

test_data.head()
y = train_data['Survived']



features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score





def get_accuracy(X_train, X_val, y_train, y_val, max_depth=5, n_estimators=100, max_leaf_nodes=None):

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=1, max_leaf_nodes=max_leaf_nodes)

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    return accuracy_score(predictions, y_val)
depths = [5, 10, 50, 100, 500]

best_accuracy, best_depth = -float('inf'), 0

for depth in depths:

    acc = get_accuracy(X_train, X_val, y_train, y_val, depth)

    print(f"Depth: {depth} \t Accuracy {acc}")

    if acc > best_accuracy:

        best_accuracy, best_depth = acc, depth

        

print(f"Best accuracy is {best_accuracy} at depth {best_depth}")
num_leaf_nodes = [5, 10, 50, 100, 500, None]

best_accuracy, best_max_leaf_nodes = -float('inf'), None

for leaf_nodes in num_leaf_nodes:

    acc = get_accuracy(X_train, X_val, y_train, y_val, max_depth=best_depth, max_leaf_nodes=leaf_nodes)

    print(f"Max Leaf Nodes: {leaf_nodes} \t Accuracy: {acc}")

    if acc > best_accuracy:

        best_accuracy, best_max_leaf_nodes = acc, leaf_nodes

        

print(f"Best accuracy is {best_accuracy} at Max Leaf Nodes {best_max_leaf_nodes}")
n_estimator_trees = [100, 200, 300, 400, 500]

best_accuracy, best_n_estimator = -float('inf'), 0

for n_trees in n_estimator_trees:

    acc = get_accuracy(X_train, X_val, y_train, y_val, max_depth=best_depth, max_leaf_nodes=best_max_leaf_nodes,

                      n_estimators=n_trees)

    print(f"No. of trees: {n_trees} \t Accuracy: {acc}")

    if acc > best_accuracy:

        best_accuracy, best_n_estimator = acc, n_trees

        

print(f"Best accuracy is {best_accuracy} at {best_n_estimator} no. of trees")
X_test.isnull().any()
# assigning median to null values in Fare in X_test

X_test.fillna(X_test.median(), inplace=True)
X_test.isnull().any()
final_model = RandomForestClassifier(n_estimators=best_n_estimator, max_leaf_nodes=best_max_leaf_nodes,

                                    max_depth=best_depth, random_state=1)

final_model.fit(X, y)

predictions = final_model.predict(X_test)



accuracy_score(final_model.predict(X), y) # just getting an idea
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submissions.csv', index=False)

print('Yo, it\'s done')