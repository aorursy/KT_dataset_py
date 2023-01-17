import numpy as np

import pandas as pd

import sklearn

from sklearn import preprocessing

from sklearn.neural_network import MLPClassifier

from sklearn import datasets

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix

from sklearn import tree

from sklearn.model_selection import cross_validate
# base de treinamento

adult = pd.read_csv('/kaggle/input/adult-pmr3508/train_data.csv',

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],na_values = "?")
# base de teste

testAdult = pd.read_csv('/kaggle/input/adult-pmr3508/test_data.csv', 

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"], na_values = "?")
nadult = adult.dropna() #remove missing values
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss", "Country"]]
Yadult = numAdult.Target
nTestAdult = testAdult.dropna()
numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss", "Country"]]
# Iniciando o MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
scores1 = cross_validate(mlp, Xadult, Yadult, cv = 5,scoring=('accuracy', 'f1_macro', 'precision_macro',

                                                      'recall_macro'), return_train_score=True)
# acurácia do método de multi-layer perceptrons

np.mean(scores1['train_accuracy'])
# f1_score do método de multi-layer perceptrons

np.mean(scores1['train_f1_macro'])
# precisão do método de multi-layer perceptrons

np.mean(scores1['train_precision_macro'])
# recall do método de multi-layer perceptrons

np.mean(scores1['train_recall_macro'])
gnb = GaussianNB()
gnb.fit(Xadult,Yadult)
scores2 = cross_validate(gnb, Xadult, Yadult, cv = 5, scoring=('accuracy', 'f1_macro', 'precision_macro',

                                                      'recall_macro'), return_train_score=True)
# acurácia do método de gaussian naive-bayes

np.mean(scores2['train_accuracy'])
# f1_score do método de gaussian naive-bayes

np.mean(scores2['train_f1_macro'])
# precisão do método de gaussian naive-bayes

np.mean(scores2['train_precision_macro'])
# recall do método de gaussian naive-bayes

np.mean(scores2['train_recall_macro'])
mnb = MultinomialNB()
scores3 = cross_validate(mnb, Xadult, Yadult, cv = 5, scoring=('accuracy', 'f1_macro', 'precision_macro',

                                                      'recall_macro'), return_train_score=True)
# acurácia do método de multinomial naive-bayes

np.mean(scores3['train_accuracy'])
# f1_score do método de multinomial naive-bayes

np.mean(scores3['train_f1_macro'])
# precisão do método de multinomial naive-bayes

np.mean(scores3['train_precision_macro'])
# recall do método de multinomial naive-bayes

np.mean(scores3['train_recall_macro'])
tr = tree.DecisionTreeClassifier()
scores4 = cross_validate(tr, Xadult, Yadult, cv = 5, scoring=('accuracy', 'f1_macro', 'precision_macro',

                                                      'recall_macro'), return_train_score=True)
# acurácia do método de árvore de decisão

np.mean(scores4['train_accuracy'])
# f1_score do método de árvore de decisão

np.mean(scores4['train_f1_macro'])
# precisão do método de árvore de decisão

np.mean(scores4['train_precision_macro'])
# recall do método de árvore de decisão

np.mean(scores4['train_recall_macro'])