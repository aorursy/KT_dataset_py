# %load Naive\ Bayes.py

# Este exemplo carrega a base Wine da UCI, treina um classificador Naive Bayes

# usando Holdout e outro usando Validação Cruzada com 10 pastas. 



# Importa bibliotecas

import numpy as np

import urllib

from sklearn.naive_bayes import GaussianNB

from sklearn import  model_selection

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split



# Carrega uma base de dados do UCI

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

raw_data = urllib.request.urlopen(url)



# Carrega arquivo como uma matriz

dataset = np.loadtxt(raw_data, delimiter=",")



# Quantide de instâncias e atributos da base

print(dataset.shape)



# Coloca em X os 13 atributos de entrada e em y as classes

# Observe que na base Wine a classe é o primeiro atributo 

X = dataset[:,1:13]

y = dataset[:,0]



# EXEMPLO USANDO HOLDOUT

# Holdout -> dividindo a base em treinamento (70%) e teste (30%), estratificada

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3, random_state=42, stratify=y)



# Treina o classificador

clfa = GaussianNB()

clfa = clfa.fit(X_train, y_train)



# Testa usando a base de testes

predicted=clfa.predict(X_test)



# Calcula a acurácia na base de teste

score=clfa.score(X_test, y_test)



# Calcula a matriz de confusão

matrix = confusion_matrix(y_test, predicted)



# Apresenta os resultados

print("Accuracy = %.2f " % score)

print("Confusion Matrix:")

print(matrix)



# EXEMPLO USANDO VALIDAÇÃO CRUZADA

clfb = GaussianNB()

folds=10

result = model_selection.cross_val_score(clfb, X, y, cv=folds)

print("\nCross Validation Results %d folds:" % folds)

print("Mean Accuracy: %.2f" % result.mean())

print("Mean Std: %.2f" % result.std())



# Matriz de Confusão da Validação Cruzada

Z = model_selection.cross_val_predict(clfb, X, y, cv=folds)

cm=confusion_matrix(y, Z)

print("Confusion Matrix:")

print(cm)
