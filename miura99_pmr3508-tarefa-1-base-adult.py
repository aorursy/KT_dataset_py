#Primeira parte: importando bibliotecas, preparando o ambiente de trabalho
import pandas as pd
import sklearn

#importando os dados para um dataframe "sujo"
raw = pd.read_csv("../input/badult/train_data.csv",
        names= None,
        engine='python',
        na_values = '?')
#Eliminando dados faltantes:
from sklearn import preprocessing
clean = raw.dropna()
clean.info()
#Identificando número de classes dos atributos não numéricos
obg = raw[['workclass','education','marital.status','occupation','relationship','race','sex','native.country','income']]
obg.nunique()
import matplotlib.pyplot as plt
from sklearn import preprocessing
analysis = clean
analysis = analysis.apply(preprocessing.LabelEncoder().fit_transform)
plt.matshow(analysis.corr())
anl0 = analysis.corr().income.sort_values(ascending=True)
anl0
anl1 = pd.get_dummies(clean[['relationship','marital.status','capital.loss', 'sex', 'hours.per.week', 'age', 'education.num', 'capital.gain', 'income']])
anl1 = anl1.corr().loc[:,'income_>50K'].sort_values(ascending=True)
anl1
anl1_5 = pd.get_dummies(clean)
anl1_5 = anl1_5.corr().loc[:,'income_>50K'].sort_values(ascending=True).where(lambda x : abs(x) > 0.15).dropna()
anl1_5
anl2 = clean[['occupation','income','race']]
anl2 = pd.get_dummies(anl2).drop(columns = 'income_<=50K')
anl2 = anl2.corr().loc[:,'income_>50K'].sort_values(ascending=True).where(lambda x : abs(x) > 0.088).dropna()
anl2
#Definindo base de treino com as variáveis de maior correlação e usando cross-validation:
train_clean = pd.get_dummies(clean)
index = anl1.where(lambda x : abs(x) > 0.07).dropna().index[1:-1].append(anl2.index[:-1])
#preparando base de teste:
test_raw = pd.read_csv("../input/badult/test_data.csv",
        names= None,
        engine='python')
X_train = train_clean[index].drop(columns='sex_Female')
Y_train = train_clean.loc[:,'income_>50K']
test_clean = pd.get_dummies(test_raw)
test_clean = test_clean.dropna()
X_test = test_clean[index].drop(columns='sex_Female')
X_test.info()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
#usarei gridsearch para verificar os melhores hiperparâmetros do kNN.
k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']
p_options = list(range(1,3))
param_grid = dict(n_neighbors=k_range, p=p_options)#, p=p_options
#inicializando o classificador a ser verificado no GridSearchCV
knn = KNeighborsClassifier(n_neighbors=5)

from sklearn.model_selection import cross_val_score
#inicializando o GridSearchCV
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_jobs = -2)  
grid.fit(X_train, Y_train)
print(grid.best_estimator_)
print(grid.best_score_)
#Criando o melhor kNN encontrado (para o caso, k = 16, p = 1):
f_kNN = grid.best_estimator_
f_kNN.fit(X_train,Y_train)
Y_test = f_kNN.predict(X_test)
Y_test_copy = Y_test
Y_test_copy = Y_test_copy.tolist()
answer = [["Id","income"]]
for output in range(len(Y_test_copy)):
    if Y_test_copy[output] == 0:
        Y_test_copy[output] = '<=50K'
    else:
        Y_test_copy[output] = '>50K'
    answer.append([output,Y_test_copy[output]])

import csv
myFile = open("submit.csv", 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(answer)
