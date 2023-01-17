import os

import numpy as np 
import pandas as pd 
import random as rnd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
df_dataSet = pd.read_csv('../input/iriscsv/iris.arff.csv')
df_dataSet.head()
df_dataSet.tail()
df_dataSet.info()
df_dataSet.describe()
df_dataSet.describe(include=['O'])
g = sns.FacetGrid(df_dataSet, col='class')
g.map(plt.hist, 'sepallength')
g = sns.FacetGrid(df_dataSet, col='class')
g.map(plt.hist, 'sepalwidth')
g = sns.FacetGrid(df_dataSet, col='class')
g.map(plt.hist, 'petallength')
g = sns.FacetGrid(df_dataSet, col='class')
g.map(plt.hist, 'petalwidth')
df_dataSet['classInt'] = df_dataSet['class'].map( {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3} ).astype(int)
df_dataSet['petallengthBand'] = pd.cut(df_dataSet['petallength'], 5)
df_dataSet[['petallengthBand', 'classInt']].groupby(['petallengthBand'], as_index=False).mean().sort_values(by='petallengthBand', ascending=True)
df_tr_inputData = df_dataSet.drop(['class', 'classInt', 'petallengthBand'], axis=1)
df_tr_class = df_dataSet["class"]
df_tr_inputData.head()
decision_tree = DecisionTreeClassifier()
decision_tree.fit(df_tr_inputData, df_tr_class)
acc_decision_tree = round(decision_tree.score(df_tr_inputData, df_tr_class) * 100, 2)
print('precision {} '.format(acc_decision_tree))
gaussian = GaussianNB()
gaussian.fit(df_tr_inputData, df_tr_class)
acc_gaussian = round(gaussian.score(df_tr_inputData, df_tr_class) * 100, 2)
print('precision {} '.format(acc_gaussian))
perceptron = Perceptron()
perceptron.fit(df_tr_inputData, df_tr_class)
acc_perceptron = round(perceptron.score(df_tr_inputData, df_tr_class) * 100, 2)
print('precision {} '.format(acc_perceptron))
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(df_tr_inputData, df_tr_class)
acc_knn = round(knn.score(df_tr_inputData, df_tr_class) * 100, 2)
print('precision {} '.format(acc_knn))