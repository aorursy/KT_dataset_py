import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn import tree

from sklearn import neighbors

from sklearn import ensemble

from sklearn import svm

from sklearn import gaussian_process

from sklearn import naive_bayes

from sklearn import neural_network

from sklearn.model_selection import cross_val_score

%matplotlib inline
testset = pd.read_csv("../input/test.csv")

trainset = pd.read_csv("../input/train.csv")
x = pd.concat([trainset[["bone_length", "rotting_flesh", "hair_length", "has_soul"]], pd.get_dummies(trainset["color"])], axis=1)

y = trainset[["type"]]

x_test = pd.concat([testset[["bone_length", "rotting_flesh", "hair_length", "has_soul"]], pd.get_dummies(testset["color"])], axis=1)
sns.set()

sns.pairplot(trainset[["bone_length", "rotting_flesh", "hair_length", "has_soul", "type"]], hue="type")
clfs = {}



clfs['lr'] = {'clf': linear_model.LogisticRegression(), 'name':'LogisticRegression'}

clfs['rf'] = {'clf': ensemble.RandomForestClassifier(n_estimators=750, n_jobs=-1), 'name':'RandomForest'}

clfs['tr'] = {'clf': tree.DecisionTreeClassifier(), 'name':'DecisionTree'}

clfs['knn'] = {'clf': neighbors.KNeighborsClassifier(n_neighbors=4), 'name':'kNearestNeighbors'}

clfs['svc'] = {'clf': svm.SVC(kernel="linear"), 'name': 'SupportVectorClassifier'}

clfs['nusvc'] = {'clf': svm.NuSVC(), 'name': 'NuSVC'}

clfs['linearsvc'] = {'clf': svm.LinearSVC(), 'name': 'LinearSVC'}

clfs['SGD'] = {'clf': linear_model.SGDClassifier(), 'name': 'SGDClassifier'}

clfs['GPC'] = {'clf': gaussian_process.GaussianProcessClassifier(), 'name': 'GaussianProcess'}

clfs['nb'] = {'clf': naive_bayes.GaussianNB(), 'name':'GaussianNaiveBayes'}

clfs['bag'] = {'clf': ensemble.BaggingClassifier(neighbors.KNeighborsClassifier(), max_samples=0.5, max_features=0.5), 'name': "BaggingClassifier"}

clfs['gbc'] = {'clf': ensemble.GradientBoostingClassifier(), 'name': 'GradientBoostingClassifier'}

clfs['mlp'] = {'clf': neural_network.MLPClassifier(hidden_layer_sizes=(10,8,3), alpha=1e-5, solver='lbfgs'), 'name': 'MultilayerPerceptron'}
for clf in clfs:

    clfs[clf]['score'] = cross_val_score(clfs[clf]['clf'], x, y.values.ravel(), cv=100)

    print(clfs[clf]['name'] + ": %0.4f (+/- %0.4f)" % (clfs[clf]['score'].mean(), clfs[clf]['score'].std()*2))
for clf in clfs:

    clfs[clf]['clf'].fit(x, y.values.ravel())
for clf in clfs:

    clfs[clf]['predictions'] = clfs[clf]['clf'].predict(x_test)
for clf in clfs:

    sub = pd.DataFrame(clfs[clf]['predictions'])

    pd.concat([testset["id"],sub], axis=1).rename(columns = {0: 'type'}).to_csv("submission_" + clfs[clf]['name'] + ".csv", index=False)