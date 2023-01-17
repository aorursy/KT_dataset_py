from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn import model_selection

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.feature_selection import RFE

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegressionCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import ExtraTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import BaggingClassifier

from xgboost import XGBClassifier

from pandas import read_csv

from pandas import DataFrame

from pandas.plotting import scatter_matrix

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
# Carregando os dados

dados = read_csv("../input/dataset_treino.csv")

dados_teste = read_csv("../input/dataset_teste.csv")
# Renomeando as colunas, para uma melhor legibilidade

dados.columns = ['id', 'preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

dados_teste.columns = ['id', 'preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']
# Removendo a coluna id

dados.drop(['id'], axis = 1, inplace = True)

dados_teste.drop(['id'], axis = 1, inplace = True)
# Visualizando as primeiras 20 linhas

dados.head(20)
# Visualizando as dimensões

dados.shape
# Tipo de dados de cada atributo

dados.dtypes
# Colunas com valores nulos

dados.isnull().any()
# Sumário estatístico

dados.describe()
# Distribuição das classes

dados.groupby('class').size()
# Correlação de Pearson

corr = dados.corr(method = 'pearson')

corr
# Verificando o skew de cada atributo

dados.skew()
# Histograma

dados.hist(bins = 50, figsize = (20, 15))

plt.show()
# Density Plot Univariado

dados.plot(kind = 'density', subplots = True, layout = (3,3), sharex = False)

plt.show()
# Density Plot usando Seaborn

fig, ax = plt.subplots(4,2, figsize = (16,16))

sns.distplot(dados.preg, bins = 20, ax = ax[0,0]) 

sns.distplot(dados.plas, bins = 20, ax = ax[0,1]) 

sns.distplot(dados.pres, bins = 20, ax = ax[1,0]) 

sns.distplot(dados.skin, bins = 20, ax = ax[1,1]) 

sns.distplot(dados.test, bins = 20, ax = ax[2,0])

sns.distplot(dados.mass, bins = 20, ax = ax[2,1])

sns.distplot(dados.pedi, bins = 20, ax = ax[3,0]) 

sns.distplot(dados.age, bins = 20, ax = ax[3,1]) 
# Box and Whisker Plots

dados.plot(kind = 'box', subplots = True, layout = (3,3), sharex = False, sharey = False)

plt.show()
# Matriz de Correlação

correlations = dados.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin = -1, vmax = 1)

fig.colorbar(cax)

ticks = np.arange(0, 9, 1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(dados.columns)

ax.set_yticklabels(dados.columns)

plt.show()
# Correlação - Seaborn

sns.heatmap(corr, annot = True)
# Correlação - Seaborn

# sns.set(font_scale=1.15)

plt.figure(figsize=(14, 10))



sns.heatmap(corr, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='YlGnBu',linecolor="black")

plt.title('Correlação entre as variáveis');
# Scatter Plot 

scatter_matrix(dados)

plt.show()
# Pairplot

sns.pairplot(dados)
# Boxplot com orientação vertical

sns.boxplot(data = dados, orient = "v")
max_skinthickness = dados['skin'].max()

dados = dados[dados['skin'] != max_skinthickness]
# Calcula o valor mediano para "plas" e substitui na coluna do dataset onde os valores são 0



# Dados de treino

median = dados['plas'].median()

dados['plas'] = dados['plas'].replace(to_replace = 0, value = median)



# Dados de teste

median = dados_teste['plas'].median()

dados_teste['plas'] = dados_teste['plas'].replace(to_replace = 0, value = median)
# Calcula o valor mediano para "pres" e substitui na coluna do dataset onde os valores são 0 



# Dados de treino

median = dados['pres'].median()

dados['pres'] = dados['pres'].replace(to_replace = 0, value = median)



# Dados de teste

median = dados_teste['pres'].median()

dados_teste['pres'] = dados_teste['pres'].replace(to_replace = 0, value = median)
# Calcula o valor mediano para "skin" e substitui na coluna do dataset onde os valores são 0



# Dados de treino

median = dados['skin'].median()

dados['skin'] = dados['skin'].replace(to_replace = 0, value = median)



# Dados de teste

median = dados_teste['skin'].median()

dados_teste['skin'] = dados_teste['skin'].replace(to_replace = 0, value = median)
# Calcula o valor mediano para "test" e substitui na coluna do dataset onde os valores são 0 



# Dados de treino

median = dados['test'].median()

dados['test'] = dados['test'].replace(to_replace = 0, value = median)



# Dados de teste

median = dados_teste['test'].median()

dados_teste['test'] = dados_teste['test'].replace(to_replace = 0, value = median)
# Calcula o valor mediano para "mass" e substitui na coluna do dataset onde os valores são 0 



# Dados de treino

median = dados['mass'].median()

dados['mass'] = dados['mass'].replace(to_replace = 0, value = median)



# Dados de teste

median = dados_teste['mass'].median()

dados_teste['mass'] = dados_teste['mass'].replace(to_replace = 0, value = median)
# Sumário estatístico

dados.describe()
# Verificando o skew de cada atributo

dados.skew()
# Density Plot usando Seaborn

fig, ax = plt.subplots(4,2, figsize = (16,16))

sns.distplot(dados.preg, bins = 20, ax = ax[0,0]) 

sns.distplot(dados.plas, bins = 20, ax = ax[0,1]) 

sns.distplot(dados.pres, bins = 20, ax = ax[1,0]) 

sns.distplot(dados.skin, bins = 20, ax = ax[1,1]) 

sns.distplot(dados.test, bins = 20, ax = ax[2,0])

sns.distplot(dados.mass, bins = 20, ax = ax[2,1])

sns.distplot(dados.pedi, bins = 20, ax = ax[3,0]) 

sns.distplot(dados.age, bins = 20, ax = ax[3,1]) 
# Ajuste nos dados de treino e teste

array = dados.values

array_test = dados_teste.values



# Variáveis independentes

X_treino = array[:,0:8]

X_teste = array_test



# Variável dependente

y_treino = array[:,8]
# Seleção Univariada

# Extração de Variáveis com Testes Estatísticos Univariados (Teste qui-quadrado)



# Extração de Variáveis

test = SelectKBest(score_func = chi2, k = 4)

fit = test.fit(X_treino, y_treino)



# Sumarizando o score

print(fit.scores_)

features = fit.transform(X_treino)



# Sumarizando atributos selecionados

print(dados.columns[0:8])

print(features[0:1,:])
# Eliminação Recursiva de Variáveis



# Criação do modelo

modelo = LogisticRegression()



# RFE

rfe = RFE(modelo, 4)

fit = rfe.fit(X_treino, y_treino)



# Print dos resultados

print("Número de Atributos: %d" % fit.n_features_)

print(dados.columns[0:8])

print("Atributos Selecionados: %s" % fit.support_)

print("Ranking dos Atributos: %s" % fit.ranking_)
# Importância do Atributo com o Extra Trees Classifier



# Criação do Modelo - Feature Selection

modelo = ExtraTreesClassifier()

modelo.fit(X_treino, y_treino)



# Print dos Resultados

print(dados.columns[0:8])

print(modelo.feature_importances_)
# Removendo as colunas pres e skin

dados.drop(['pres', 'skin'], axis = 1, inplace = True)

dados_teste.drop(['pres', 'skin'], axis = 1, inplace = True)
# Visualizando as primeiras 20 linhas - dados de treino

dados.head(20)
# Visualizando as dimensões - dados de treino

dados.shape
# Visualizando as primeiras 20 linhas - dados de teste

dados_teste.head(20)
# Visualizando as dimensões - dados de teste

dados_teste.shape
# Ajuste nos dados de treino e teste

array = dados.values

array_teste = dados_teste.values



# Variáveis independentes

X_treino = array[:,0:6]

X_teste = array_teste



# Variável dependente

y_treino = array[:,6]
# Criando modelo de Machine Learning a partir de cada algoritmo

# Vamos utilizar como métrica a acurácia. Quanto maior o valor, melhor.

modelos = []

modelos.append(('LR', LogisticRegression()))

modelos.append(('LRCV', LogisticRegressionCV()))

modelos.append(('LDA', LinearDiscriminantAnalysis()))

modelos.append(('KNN', KNeighborsClassifier()))

modelos.append(('DT', DecisionTreeClassifier()))

modelos.append(('ET', ExtraTreeClassifier()))

modelos.append(('NB', GaussianNB()))

modelos.append(('SVM', SVC()))

resultados = []

nomes = []



# Percorrendo cada um dos modelos

for nome, modelo in modelos:

    kfold = model_selection.KFold(n_splits = 20, random_state = 7)

    cross_val_result = model_selection.cross_val_score(modelo, X_treino, 

                                                        y_treino, 

                                                        cv = kfold, 

                                                        scoring = 'accuracy')

    resultados.append(cross_val_result)

    nomes.append(nome)

    texto = "%s: %f (%f)" % (nome, cross_val_result.mean(), cross_val_result.std())

    print(texto)
# Comparando os algoritmos

fig = plt.figure(figsize = (12,8))

fig.suptitle('Comparando os Algoritmos')

ax = fig.add_subplot(111)

plt.boxplot(resultados)

ax.set_xticklabels(nomes)

plt.show()
# Aplicando Padronização ao conjunto de dados - StandardScaler

pipelines = []

pipelines.append(('Scaled-LR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())])))

pipelines.append(('Scaled-LRCV', Pipeline([('Scaler', StandardScaler()),('LRCV', LogisticRegressionCV())])))

pipelines.append(('Scaled-LDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))

pipelines.append(('Scaled-KNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))

pipelines.append(('Scaled-DT', Pipeline([('Scaler', StandardScaler()),('DT', DecisionTreeClassifier())])))

pipelines.append(('Scaled-ET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreeClassifier())])))

pipelines.append(('Scaled-NB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))

pipelines.append(('Scaled-SVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))

resultados = []

nomes = []



# Percorrendo cada um dos modelos

for nome, modelo in pipelines:

    kfold = model_selection.KFold(n_splits = 20, random_state = 7)

    cross_val_result = model_selection.cross_val_score(modelo, 

                                                     X_treino, 

                                                     y_treino, 

                                                     cv = kfold, 

                                                     scoring = 'accuracy')

    resultados.append(cross_val_result)

    nomes.append(nome)

    texto = "%s: %f (%f)" % (nome, cross_val_result.mean(), cross_val_result.std())

    print(texto)
# Comparando os algoritmos

fig = plt.figure(figsize = (12,8))

fig.suptitle('Comparando os Algoritmos')

ax = fig.add_subplot(111)

plt.boxplot(resultados)

ax.set_xticklabels(nomes)

plt.show()
# Aplicando Padronização ao conjunto de dados - MinMaxScaler

pipelines = []

pipelines.append(('Scaled-LR', Pipeline([('Scaler', MinMaxScaler()),('LR', LogisticRegression())])))

pipelines.append(('Scaled-LRCV', Pipeline([('Scaler', MinMaxScaler()),('LRCV', LogisticRegressionCV())])))

pipelines.append(('Scaled-LDA', Pipeline([('Scaler', MinMaxScaler()),('LDA', LinearDiscriminantAnalysis())])))

pipelines.append(('Scaled-KNN', Pipeline([('Scaler', MinMaxScaler()),('KNN', KNeighborsClassifier())])))

pipelines.append(('Scaled-DT', Pipeline([('Scaler', MinMaxScaler()),('DT', DecisionTreeClassifier())])))

pipelines.append(('Scaled-ET', Pipeline([('Scaler', MinMaxScaler()),('ET', ExtraTreeClassifier())])))

pipelines.append(('Scaled-NB', Pipeline([('Scaler', MinMaxScaler()),('NB', GaussianNB())])))

pipelines.append(('Scaled-SVM', Pipeline([('Scaler', MinMaxScaler()),('SVM', SVC())])))

resultados = []

nomes = []



# Percorrendo cada um dos modelos

for nome, modelo in pipelines:

    kfold = model_selection.KFold(n_splits = 20, random_state = 7)

    cross_val_result = model_selection.cross_val_score(modelo, 

                                                     X_treino, 

                                                     y_treino, 

                                                     cv = kfold, 

                                                     scoring = 'accuracy')

    resultados.append(cross_val_result)

    nomes.append(nome)

    texto = "%s: %f (%f)" % (nome, cross_val_result.mean(), cross_val_result.std())

    print(texto)
# Comparando os algoritmos

fig = plt.figure(figsize = (12,8))

fig.suptitle('Comparando os Algoritmos')

ax = fig.add_subplot(111)

plt.boxplot(resultados)

ax.set_xticklabels(nomes)

plt.show()
# Embora o KNN tenha apresentado a menor taxa de erro após a padronização dos dados, podemos ainda otimizá-lo

# com o ajuste dos parâmetros.



# Definindo a escala

scaler = StandardScaler().fit(X_treino)

rescaledX = scaler.transform(X_treino)



# Ajustando valores para o tamanho do K

neighbors = [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 19, 21]

algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']

weights = ['uniform','distance']

leaf_size = [1, 2, 3]

p = [1, 2]

valores_grid = dict(n_neighbors = neighbors,

                    algorithm = algorithm,

                    weights = weights, 

                    leaf_size = leaf_size,

                    p = p)



# Criando o modelo

modelo = KNeighborsClassifier()



# Definindo K

kfold = model_selection.KFold(n_splits = 20, random_state = 7)



# Testando diferentes combinações com os valores de K

grid = model_selection.GridSearchCV(estimator = modelo, param_grid = valores_grid, cv = kfold, scoring = 'accuracy',return_train_score=True)

grid_result = grid.fit(rescaledX, y_treino)



# Criando uma lista com média dos scores

mean_score = [mean for mean in grid_result.cv_results_['mean_test_score']]



# Criando uma lista com desvio padrão dos scores

std_score = [std for std in grid_result.cv_results_['std_test_score']]



# Criando uma lista com os parametros utilizados

params = [params for params in grid_result.cv_results_['params']]



# Variável para indicar a linha da lista

i = 0



print("Melhor Acurácia: %f utilizando %s" % (grid_result.best_score_, grid_result.best_params_))

for param in params:

    print("%f (%f) with: %r" % (mean_score[i],std_score[i],param))

    i = i +1;
# Vamos agora ajustar os parâmeros do SVM.



# Definindo a escala

scaler = StandardScaler().fit(X_treino)

rescaledX = scaler.transform(X_treino)



# Ajustando os parâmetros do SVM:

# c_values - indica o nível das margens dos vector machines

# kernel_values - tipos de kernel usados no SVM

# Faremos diferentes combinações desses métodos, a fim de verificar qual é a melhor combinação.

c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]

kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']

valores_grid = dict(C = c_values, kernel = kernel_values)



# Criando o modelo

modelo = SVC()



# Definindo K

kfold = model_selection.KFold(n_splits = 20, random_state = 7)



# Testando diferentes combinações com os parâmetros

grid = model_selection.GridSearchCV(estimator = modelo, param_grid = valores_grid, cv = kfold, scoring = 'accuracy',return_train_score=True)

grid_result = grid.fit(rescaledX, y_treino)



# Criando uma lista com média dos scores

mean_score = [mean for mean in grid_result.cv_results_['mean_test_score']]



# Criando uma lista com desvio padrão dos scores

std_score = [std for std in grid_result.cv_results_['std_test_score']]



# Criando uma lista com os parametros utilizados

params = [params for params in grid_result.cv_results_['params']]



# Variável para indicar a linha da lista

i = 0



print("Melhor Acurácia: %f utilizando %s" % (grid_result.best_score_, grid_result.best_params_))

for param in params:

    print("%f (%f) with: %r" % (mean_score[i],std_score[i],param))

    i = i +1;
ensembles = []

ensembles.append(('AB', AdaBoostClassifier()))

ensembles.append(('GB', GradientBoostingClassifier()))

ensembles.append(('RF', RandomForestClassifier()))

ensembles.append(('ET', ExtraTreesClassifier()))

ensembles.append(('BG', BaggingClassifier()))

ensembles.append(('XG', XGBClassifier()))

resultados = []

nomes = []



# Percorrendo cada um dos modelos

for nome, modelo in ensembles:

    kfold = model_selection.KFold(n_splits = 20, random_state = 7)

    cross_val_result = model_selection.cross_val_score(modelo, 

                                                        X_treino, 

                                                        y_treino, 

                                                        cv = kfold, 

                                                        scoring = 'accuracy')

    resultados.append(cross_val_result)

    nomes.append(nome)

    texto = "%s: %f (%f)" % (nome, cross_val_result.mean(), cross_val_result.std())

    print(texto)
# Comparando os algoritmos

fig = plt.figure(figsize = (12,8))

fig.suptitle('Comparando Algoritmos Ensemble')

ax = fig.add_subplot(111)

plt.boxplot(resultados)

ax.set_xticklabels(nomes)

plt.show()
# Aplicando Padronização ao conjunto de dados - StandardScaler

pipelines = []

pipelines.append(('Scaled-AB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostClassifier())])))

pipelines.append(('Scaled-GB', Pipeline([('Scaler', StandardScaler()),('GB', GradientBoostingClassifier())])))

pipelines.append(('Scaled-RF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestClassifier())])))

pipelines.append(('Scaled-ET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesClassifier())])))

pipelines.append(('Scaled-BG', Pipeline([('Scaler', StandardScaler()),('BG', BaggingClassifier())])))

pipelines.append(('Scaled-XG', Pipeline([('Scaler', StandardScaler()),('XG', XGBClassifier())])))

resultados = []

nomes = []



# Percorrendo cada um dos modelos

for nome, modelo in pipelines:

    kfold = model_selection.KFold(n_splits = 20, random_state = 7)

    cross_val_result = model_selection.cross_val_score(modelo, 

                                                     X_treino, 

                                                     y_treino, 

                                                     cv = kfold, 

                                                     scoring = 'accuracy')

    resultados.append(cross_val_result)

    nomes.append(nome)

    texto = "%s: %f (%f)" % (nome, cross_val_result.mean(), cross_val_result.std())

    print(texto)
# Comparando os algoritmos

fig = plt.figure(figsize = (12,8))

fig.suptitle('Comparando Algoritmos Ensemble')

ax = fig.add_subplot(111)

plt.boxplot(resultados)

ax.set_xticklabels(nomes)

plt.show()
# Aplicando Padronização ao conjunto de dados - MinMaxScaler

pipelines = []

pipelines.append(('Scaled-AB', Pipeline([('Scaler', MinMaxScaler()),('AB', AdaBoostClassifier())])))

pipelines.append(('Scaled-GB', Pipeline([('Scaler', MinMaxScaler()),('GB', GradientBoostingClassifier())])))

pipelines.append(('Scaled-RF', Pipeline([('Scaler', MinMaxScaler()),('RF', RandomForestClassifier())])))

pipelines.append(('Scaled-ET', Pipeline([('Scaler', MinMaxScaler()),('ET', ExtraTreesClassifier())])))

pipelines.append(('Scaled-BG', Pipeline([('Scaler', MinMaxScaler()),('BG', BaggingClassifier())])))

pipelines.append(('Scaled-XG', Pipeline([('Scaler', MinMaxScaler()),('XG', XGBClassifier())])))

resultados = []

nomes = []



# Percorrendo cada um dos modelos

for nome, modelo in pipelines:

    kfold = model_selection.KFold(n_splits = 20, random_state = 7)

    cross_val_result = model_selection.cross_val_score(modelo, 

                                                     X_treino, 

                                                     y_treino, 

                                                     cv = kfold, 

                                                     scoring = 'accuracy')

    resultados.append(cross_val_result)

    nomes.append(nome)

    texto = "%s: %f (%f)" % (nome, cross_val_result.mean(), cross_val_result.std())

    print(texto)
# Comparando os algoritmos

fig = plt.figure(figsize = (12,8))

fig.suptitle('Comparando Algoritmos Ensemble')

ax = fig.add_subplot(111)

plt.boxplot(resultados)

ax.set_xticklabels(nomes)

plt.show()
# Vamos agora ajustar os parâmeros do Ada Boost Classifier.



# Definindo a escala

scaler = StandardScaler().fit(X_treino)

rescaledX = scaler.transform(X_treino)



# Ajustando os parâmetros:

n_estimators = [50, 100, 200, 500]

algorithm = ['SAMME', 'SAMME.R']

learning_rate = [0.01, 0.020, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5, 1]

valores_grid = dict(n_estimators = n_estimators,

                    algorithm = algorithm,

                    learning_rate = learning_rate)



# Estimador

# O estimador default é o DecisionTreeClassifier(max_depth=1)

estimador = DecisionTreeClassifier(max_depth=2)



# Criando o modelo

modelo = AdaBoostClassifier(estimador)



# Definindo K

kfold = model_selection.KFold(n_splits = 20, random_state = 7)



# Testando diferentes combinações com os parâmetros

grid = model_selection.GridSearchCV(estimator = modelo, param_grid = valores_grid, cv = kfold, scoring = 'accuracy',return_train_score=True)

grid_result = grid.fit(rescaledX, y_treino)



# Criando uma lista com média dos scores

mean_score = [mean for mean in grid_result.cv_results_['mean_test_score']]



# Criando uma lista com desvio padrão dos scores

std_score = [std for std in grid_result.cv_results_['std_test_score']]



# Criando uma lista com os parametros utilizados

params = [params for params in grid_result.cv_results_['params']]



# Variável para indicar a linha da lista

i = 0



print("Melhor Acurácia: %f utilizando %s" % (grid_result.best_score_, grid_result.best_params_))

for param in params:

    print("%f (%f) with: %r" % (mean_score[i],std_score[i],param))

    i = i + 1;
# Usando esse aqui por enquanto para submeter resultados para o Kaggle!!

# Preparando a versão final do modelo

scaler = StandardScaler().fit(X_treino)

rescaledX = scaler.transform(X_treino)

dt = DecisionTreeClassifier(max_depth=2)

modelo = AdaBoostClassifier(base_estimator = dt, n_estimators=500, algorithm='SAMME', learning_rate=0.01)

modelo.fit(rescaledX, y_treino)
# Aplicando o modelo aos dados de teste

scaler = StandardScaler().fit(X_teste)

rescaledValidationX = scaler.transform(X_teste)

previsoes = modelo.predict(rescaledValidationX)
# Submissão

test = read_csv("../input/dataset_teste.csv")



submission = DataFrame()

submission['id'] = test["id"]

submission['classe'] = previsoes.astype(int)



submission.to_csv('submission.csv', index = False)
submission.head(20)