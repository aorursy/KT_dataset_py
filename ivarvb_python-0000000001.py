# -*- coding: utf-8 -*-

"""
Exemplo de ciência de dados e visualização com python
Autor: Ivar Vargas Belizario
E-mail: ivar@usp.br
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

"""
===================================================
I. Ciência de dados
===================================================

===================================================
1. Leitura dos datos para o treino e para o teste
===================================================
"""

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

"""
===================================================
2. Pre-processamento
===================================================

===================================================
2.1 Limpeza e amostragem
===================================================
"""
    
# Definir as colunas da etiqueta da classe (target) e do identificador (id)
column_target = 'Target'
column_id = 'Id'

# limpar os atributos que apresentam valores nulls
data = train.dropna(axis='columns')

# número de instancias antes da amostragem
print ("Total data: ",len(data))

# porcentagem para a amostragem
c_sample = 0.1999

# separação dos atributos: identificador da instancia (id)
# dos atributos data (X) e do atributo que contem a etiqueta da classe (y)
X = data
y = data[column_target]

X_null, X_train, y_null, y_train = train_test_split(X, y, test_size=c_sample, random_state=0)

ID = X_train[column_id]
y = X_train[column_target]
X = X_train.drop([column_id, column_target], axis=1).select_dtypes(include=[np.number])

train_select_atributes = X.columns

X = X.values
y = y.values

print ("Amostragem: ",len(X))

"""
===================================================
2. Processamento
===================================================

===================================================
2.1 Redução da dimensionalidade (feature selection)
===================================================
"""
a=3





"""
===================================================
3. Modelo de aprendisagem (aprendisagem supervisionado):
===================================================

===================================================
3.1. Treinamento:
===================================================
"""

# definir o modelo para a classificação
model = RandomForestClassifier(random_state=0, n_estimators=100)

# modelo de treinamento com k-fold (10-fold)
kf = StratifiedKFold(n_splits=10)
outcomes = []

# para cada fold
for train_index, test_index in kf.split(X, y):
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    model.fit(Xtrain, ytrain)
    expected = ytest
    predictions = model.predict(Xtest)

    accuracy = accuracy_score(ytest, predictions)
    outcomes.append(accuracy)

# imprimir a media da acuracia obtida no treinamento
mean_outcome = np.array(outcomes).mean()

print ("Mean Accuracy:", mean_outcome)

"""
===================================================
3.2. Teste:
===================================================
"""

# selecão de atributos igual ao feito con o conjunto de treino
X_test = test[train_select_atributes]
x_test_id = test[column_id]
predictions = model.predict(X)

result = "Id,Target\n"
for i in range(len (predictions) ):
    result+=str(x_test_id[i])+","+str(predictions[i])+"\n"

"""
# salvar resultados obtidos do conjunto de dados de teste
f = open("../input/result.csv", "w")
f.write(result)
f.close() 
"""
print (result)

"""
===================================================
II. Visualização do conjunto de dados (projeções)
===================================================
"""

#print (y)
isfineClass = False
for i in range(len(y)):
    if y[i]==0:
        isfineClass=True
        break;

if isfineClass==False:
    for i in range(len(y)):
        v = y[i]
        y[i] = v-1
            
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)
#print (X_2d)         
target_ids = range(len(y))

plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'

for i in range(len(y)):
    v = y[i]
    plt.plot(X_2d[i, 0], X_2d[i, 1], 'o', color=colors[v])
    #plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
#plt.legend()
plt.show()

