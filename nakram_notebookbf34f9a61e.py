# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn.tree as tree



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print(train.shape)

print(test.shape)

# Any results you write to the current directory are saved as output.



# Create the target and features numpy arrays: target, features_one



#train['Age'].fillna(train['Age'].median())



#print(train['Age'])

train['Age'] = train['Age'].fillna(train['Age'].median())

#print(train['Age'])



target = train["Survived"].values

features_one = train[["Pclass", "Sex", "Age", "Fare"]].values



# Fit your first decision tree: my_tree_one

my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one, target)



# Look at the importance and score of the included features

print(my_tree_one.feature_importances_)

print(my_tree_one.score(features_one, target))







test_features = test[["Pclass", "Sex", "Age", "Fare"]].values



my_prediction = my_tree_one.predict(test_features)



PassengerId = np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

print(my_solution)



# Check that your data frame has 418 entries

print(my_solution.shape)



# Write your solution to a csv file with the name my_solution.csv

#my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])