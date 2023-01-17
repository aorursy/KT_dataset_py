import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import neural_network

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

%matplotlib inline
testset = pd.read_csv("../input/test.csv")

trainset = pd.read_csv("../input/train.csv")
sns.set()

sns.pairplot(trainset[["bone_length", "rotting_flesh", "hair_length", "has_soul", "type"]], hue="type")
trainset['hair_soul'] = trainset.apply(lambda row: row['hair_length']*row['has_soul'],axis=1)

testset['hair_soul'] = testset.apply(lambda row: row['hair_length']*row['has_soul'],axis=1)
x = trainset[["bone_length", "rotting_flesh", "hair_length", "has_soul"]]

x_hair_soul = trainset[["bone_length", "rotting_flesh", "hair_length", "has_soul", "hair_soul"]]



x_test = testset[["bone_length", "rotting_flesh", "hair_length", "has_soul"]]

x_test_hair_soul = testset[["bone_length", "rotting_flesh", "hair_length", "has_soul", "hair_soul"]]



y = trainset[["type"]]
parameters = {'solver': ['lbfgs'], 'max_iter': [500,1000,1500], 'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':np.arange(5, 12), 'random_state':[0,1,2,3,4,5,6,7,8,9]}

clf_grid = GridSearchCV(neural_network.MLPClassifier(), parameters, n_jobs=-1)



parameters = {'solver': ['lbfgs'], 'max_iter': [500,1000,1500], 'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':np.arange(5, 12), 'random_state':[0,1,2,3,4,5,6,7,8,9]}

clf_grid_hair_soul = GridSearchCV(neural_network.MLPClassifier(), parameters, n_jobs=-1)
clf_grid.fit(x,y.values.ravel())

clf_grid_hair_soul.fit(x_hair_soul,y.values.ravel())



print("-----------------Original Features--------------------")

print("Best score: %0.4f" % clf_grid.best_score_)

print("Using the following parameters:")

print(clf_grid.best_params_)

print("-----------------Added hair_soul feature--------------")

print("Best score: %0.4f" % clf_grid_hair_soul.best_score_)

print("Using the following parameters:")

print(clf_grid.best_params_)

print("------------------------------------------------------")
clf = neural_network.MLPClassifier(alpha=0.1, hidden_layer_sizes=(6), max_iter=500, random_state=3, solver='lbfgs')

clf.fit(x, y.values.ravel())
preds = clf.predict(x_test)
sub = pd.DataFrame(preds)

pd.concat([testset["id"],sub], axis=1).rename(columns = {0: 'type'}).to_csv("submission_neural_net.csv", index=False)