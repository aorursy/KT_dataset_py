import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import pandas as pd
nRowsRead = None 

data = pd.read_csv('/kaggle/input/abalone.csv', delimiter=',', nrows = nRowsRead)

data.dataframeName = 'abalone.csv'

nRow, nCol = data.shape

print(f'There are {nRow} rows and {nCol} columns')
data.isnull().sum()
data = data.sample(n = 1000)
data['Age'] = data['Rings']+1.5  # l'età corrisponde al numero di anelli +1.5

data.drop('Rings', axis = 1, inplace = True)

data.head(10)
data = pd.get_dummies(data)



ranges = (0, 8.5, 14.5, 30) 

classes_names = ['young', 'adult', 'old']

data['Age'] = pd.cut(data['Age'], bins = ranges, labels = classes_names)



data.head(10)





from sklearn.model_selection import train_test_split

from sklearn import tree



train, test = train_test_split(data, test_size = 0.2)



train_X = train.drop('Age', axis = 1)

test_X = test.drop('Age', axis = 1)

train_y = train['Age']

test_y = test['Age']



depths = [1, 3, 5, 8, 10, 20, 40]



for d in depths:

    print("\n")

    t = tree.DecisionTreeClassifier(max_depth = d, random_state = 1234) # random_state deve essere sempre uguale per avere modelli uguali

    t.fit(train_X, train_y)

    print("score = ", round(t.score(test_X, test_y), 4), " con max_depth = ", d, " e effettiva profondità = ", t.tree_.max_depth)





t = tree.DecisionTreeClassifier(random_state = 1234) # random_state deve essere sempre uguale per avere modelli uguali, max_depth non impostata

t.fit(train_X, train_y)

print("score = ", round(t.score(test_X, test_y), 4), " con profondità = ", t.tree_.max_depth)
min_samples = [40, 30, 20, 10, 8, 5, 2]



for m in min_samples:

    print("\n")

    t = tree.DecisionTreeClassifier(min_samples_leaf = m, random_state = 1234) # random_state deve essere sempre uguale per avere modelli uguali

    t.fit(train_X, train_y)

    print("score = ", round(t.score(test_X, test_y), 4), " con min_samples_leaf = ", m)

def post_prune(tree, index, soglia):

    if tree.children_left[index] != -1: # bottom-up

        post_prune(tree, tree.children_left[index], soglia)

        post_prune(tree, tree.children_right[index], soglia)

    if tree.value[index].min() < soglia: 

        tree.children_left[index] = -1 # -1 = foglia

        tree.children_right[index] = -1

        

min_samples = [40, 30, 20, 10, 8, 5, 2] # uso le stesse soglie

t = tree.DecisionTreeClassifier(random_state = 1234) # random_state deve essere sempre uguale per avere modelli uguali

t.fit(train_X, train_y)

print("score prima del pruning = ", round(t.score(test_X, test_y), 4))



for m in min_samples:

    print("\n")

    post_prune(t.tree_, 0, m)

    print("score dopo pruning = " , round(t.score(test_X, test_y), 4))

    t = tree.DecisionTreeClassifier(max_depth = 3, random_state = 1234)

    t.fit(train_X, train_y)












