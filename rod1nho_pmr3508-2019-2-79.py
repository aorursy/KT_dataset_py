import pandas as pd

import matplotlib.pyplot as plt

import sklearn

import seaborn as sns

from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score
adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", skipinitialspace = True, na_values = "?")

adult.set_index('Id',inplace=True)

adult.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status', 'occupation', 

                 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country', 'income']

treino = adult.dropna()

treino
testAdult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv")

testAdult.set_index('Id',inplace=True)

teste = testAdult.dropna()

teste.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status', 'occupation', 

                 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country']

teste.shape
treino["native.country"].value_counts().plot(kind="bar")
del treino["native.country"]

del teste["native.country"]
treino


sns.set

sns.pairplot(treino, palette='dark')
xTreino = treino.iloc[:,0:14].apply(preprocessing.LabelEncoder().fit_transform)

xTreino = xTreino[['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status', 'occupation', 'relationship', 

                  'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week']]

yTreino = treino.income



xTeste = teste.iloc[:,0:14].apply(preprocessing.LabelEncoder().fit_transform)
Tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=15)

xval_tree = cross_val_score(Tree, xTreino, yTreino, cv=15)

Tree.fit(xTreino, yTreino)

yPred_Tree = Tree.predict(xTeste)

accuracy_tree = xval_tree.mean()

accuracy_tree
id_index = pd.DataFrame({'Id' : list(range(len(yPred_Tree)))})

income = pd.DataFrame({'income' : yPred_Tree})

result = id_index.join(income)

result.to_csv("Tree.csv", index = False)

result
Forest = RandomForestClassifier(n_estimators=700, criterion='entropy')

xval_forest = cross_val_score(Forest, xTreino, yTreino, cv=15)

Forest.fit(xTreino, yTreino)

yPred_Forest = Forest.predict(xTeste)

accuracy_forest = xval_forest.mean()

accuracy_forest
id_index = pd.DataFrame({'Id' : list(range(len(yPred_Forest)))})

income = pd.DataFrame({'income' : yPred_Forest})

result = id_index.join(income)

result.to_csv("Forest.csv", index = False)

result
Boost = AdaBoostClassifier(n_estimators=100)

xval_boost = cross_val_score(Boost, xTreino, yTreino, cv=15)

Boost.fit(xTreino, yTreino)

yPred_Boost = Boost.predict(xTeste)

accuracy_boost = xval_boost.mean()

accuracy_boost
id_index = pd.DataFrame({'Id' : list(range(len(yPred_Boost)))})

income = pd.DataFrame({'income' : yPred_Boost})

result = id_index.join(income)

result.to_csv("Boost.csv", index = False)

result
import graphviz

graph = tree.export_graphviz(Tree, out_file=None)

teste = graphviz.Source(graph)

teste