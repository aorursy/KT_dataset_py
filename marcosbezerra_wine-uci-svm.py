# %load 14\ SVM.py

# Este exemplo carrega a base Wine da UCI, treina um classificador SVM

# usando holdout e outro usando validação cruzada com 10 pastas. 



# Importa bibliotecas

import numpy as np

import urllib

from sklearn.svm import SVC

from sklearn import  model_selection

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



# Carrega uma base de dados do UCI

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

raw_data = urllib.request.urlopen(url)



# Carrega arquivo como uma matriz

dataset = np.loadtxt(raw_data, delimiter=",")



# Imprime quantide de instâncias e atributos da base

print(dataset.shape)



# Coloca em X os 13 atributos de entrada e em y as classes

# Observe que na base Wine a classe é primeiro atributo 

X = dataset[:,1:13]

y = dataset[:,0]



# EXEMPLO USANDO HOLDOUT

# Holdout -> dividindo a base em treinamento (70%) e teste (30%), estratificada

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3, random_state=42, stratify=y)



# Treina o classificador

# Definição dos parâmetros a serem avaliados no ajuste fino do SVM

parameters = [

  {'C': [0.1, 0.5, 1, 10, 100, 500, 1000], 'kernel': ['linear']},

  {'C': [0.1, 0.5, 1, 10, 100, 500, 1000], 'gamma': [0.1, 0.001, 0.0001, 0.00001], 'kernel': ['rbf']},

 ]



clfa = SVC()

clfa = GridSearchCV(clfa, parameters, scoring = 'accuracy', cv=10, iid=False)

clfa = clfa.fit(X_train, y_train)

print(clfa.best_params_)



# testa usando a base de testes

predicted=clfa.predict(X_test)



# calcula a acurácia na base de teste

score=clfa.score(X_test, y_test)



# calcula a matriz de confusão

matrix = confusion_matrix(y_test, predicted)



# apresenta os resultados

print("Accuracy = %.2f " % score)

print("Confusion Matrix:")

print(matrix)



# EXEMPLO USANDO VALIDAÇÃO CRUZADA

clfb = SVC()

clfb = GridSearchCV(clfb, parameters, scoring = 'accuracy', cv=10, iid=False)

folds=10

result = model_selection.cross_val_score(clfb, X, y, cv=folds)

print("\nCross Validation Results %d folds:" % folds)

print("Mean Accuracy: %.2f" % result.mean())

print("Mean Std: %.2f" % result.std())



# matriz de confusão da validação cruzada

Z = model_selection.cross_val_predict(clfb, X, y, cv=folds)

cm=confusion_matrix(y, Z)

print("Confusion Matrix:")

print(cm)
