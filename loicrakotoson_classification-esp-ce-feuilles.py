import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing, model_selection, metrics, linear_model, neighbors, svm

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/Dataset_feuilles_1.csv", index_col = 0)

data.head()
Y = data.iloc[:, 0]

X = data.iloc[:,1:]
testset = pd.read_csv("../input/test.csv", index_col = 0)

testset.head()
Yle = preprocessing.LabelEncoder()

Y = Yle.fit_transform(Y)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size = 0.2)
model1 = model_selection.GridSearchCV(neighbors.KNeighborsClassifier(),

                                     {'n_neighbors': [3,5,7,10,15,20]}, cv = 10)

model1.fit(x_train, y_train)



y_pred1 = model1.predict(x_test)

print("La performance sur les données test est : ", metrics.accuracy_score(y_test, y_pred1))
result = pd.DataFrame({'KNN': round(metrics.accuracy_score(y_test, y_pred1), 4)}, index = ["accuracy"])
model2 = model_selection.GridSearchCV(svm.LinearSVC(),

                                     {'C': np.logspace(-5, 5, 7)})

model2.fit(x_train, y_train)



y_pred2 = model2.predict(x_test)

print("La performance sur les données test est : ", metrics.accuracy_score(y_test, y_pred2))
result['OVR'] = round(metrics.accuracy_score(y_test, y_pred2), 4)
model3 = model_selection.GridSearchCV(svm.SVC(),

                                     {'C': np.logspace(-5, 5, 7), 'decision_function_shape':['ovo'],

                                      'kernel':['linear']})

model3.fit(x_train, y_train)



y_pred3 = model3.predict(x_test)

print("La performance sur les données test est : ", metrics.accuracy_score(y_test, y_pred3))
result['OVO'] = round(metrics.accuracy_score(y_test, y_pred3), 4)
model4 = model_selection.GridSearchCV(svm.LinearSVC(),

                                     {'C': np.logspace(-5, 5, 7), 'multi_class':['crammer_singer']})

model4.fit(x_train, y_train)



y_pred4 = model4.predict(x_test)

print("La performance sur les données test est : ", metrics.accuracy_score(y_test, y_pred4))
result['CramSing'] = round(metrics.accuracy_score(y_test, y_pred4), 4)

modèles = {'KNN':model1, 'OVR':model2, 'OVO':model3, 'CramSing':model4} # dictionnaire des modèles
result
species = modèles[max(result)].predict(testset)

species = Yle.inverse_transform(species)

species[:5] # 5 premières valeurs
testset.insert(0, 'species', species)

testset.head()