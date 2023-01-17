# %load Classificacao.py

# Este exemplo carrega a base Wine da UCI, treina uma Arvore de Decisão usando 

# holdout e outra usando Validação Cruzada com 10 pastas. 



# Import das bibliotecas

import numpy as np

import urllib

from sklearn import tree

from sklearn import  model_selection

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.externals.six import StringIO 

from sklearn.tree import export_graphviz

from IPython.display import Image  

from IPython.display import display

import pydotplus



# Carrega uma base de dados do UCI

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

raw_data = urllib.request.urlopen(url)



# Carrega arquivo como uma matriz

dataset = np.loadtxt(raw_data, delimiter=",")



# Imprime quantidade de instâncias e atributos da base

print("Instâncias e atributos")

print(dataset.shape)



# Coloca em 'X' os 13 atributos de entrada e em 'y' as classes

# Observe que na base Wine a classe eh primeiro atributo 

X = dataset[:,1:13]

y = dataset[:,0]



# EXEMPLO USANDO HOLDOUT

# Holdout -> dividindo a base em treinamento (70%) e teste (30%), estratificada

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3, random_state=42, stratify=y)



# declara o classificador

clfa = tree.DecisionTreeClassifier(criterion='entropy')



# treina o classificador

clfa = clfa.fit(X_train, y_train)



# testa usando a base de testes

predicted=clfa.predict(X_test)



# calcula a acurácia na base de teste (taxa de acerto)

score=clfa.score(X_test, y_test)



# calcula a matriz de confusão

matrix = confusion_matrix(y_test, predicted)



# apresenta os resultados

print("\nResultados baseados em Holdout 70/30")

print("Taxa de acerto = %.2f " % score)

print("Matriz de confusao:")

print(matrix)



# EXEMPLO USANDO VALIDAÇÃO CRUZADA

clfb = tree.DecisionTreeClassifier(criterion='entropy')

folds=10

result = model_selection.cross_val_score(clfb, X, y, cv=folds)



print("\nResultados baseados em Validação Cruzada")

print("Qtde folds: %d:" % folds)

print("Taxa de Acerto: %.2f" % result.mean())

print("Desvio padrão: %.2f" % result.std())



# matriz de confusÃ£o da validacao cruzada

Z = model_selection.cross_val_predict(clfb, X, y, cv=folds)

cm=confusion_matrix(y, Z)

print("Matriz de confusão:")

print(cm)



#imprime a Arvore Gerada

print("\nÁrvore gerada no experimento baseado em Holdout")

dot_data = StringIO()

export_graphviz(clfa, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True)



graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

im=Image(graph.create_png())

display(im)
