#Bibliotecas

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')
#Pegando arquivos .csv 

df = pd.read_csv('../input/dataset_treino.csv',sep=',', encoding='ISO-8859-1')

dfTest = pd.read_csv('../input/dataset_teste.csv',sep=',', encoding='ISO-8859-1')

df.head(10)
#Shape Treino e Teste

print("Shape df Treino: ", df.shape)

print ("Shape df Teste: ", dfTest.shape)
#Data Types

print ("==== Treino ==== \n", df.dtypes)

print ("\n ==== Teste ==== \n", dfTest.dtypes)
#Colunas Com valores nulos

print("==== Treino ====\n", df.isnull().any())

print("\n ==== Teste ==== \n", dfTest.isnull().any())
#Tendo uma noção da distribuição das classes

df['classe'].value_counts()
#Bibliotecas de visualização

import matplotlib.pyplot as plt

%matplotlib inline
#Histograma

df.hist()

plt.show()
#Visualizando as correlações entre as variáveis numericamente

correlacoes = df.corr()

print (correlacoes)
#Visualizando as correlações entre as variáveis graficamente

cols = ['id', 'num_gest', 'glic', 'pres_sang','gros_pele', 'insul', 'bmi', 'ind_hist', 'idade','classe']

fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111)

cax = ax.imshow(correlacoes, interpolation='nearest', vmin = -1, vmax = 1)

fig.colorbar(cax)

ticks = np.arange(0, 10, 1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(cols)

ax.set_yticklabels(cols)

plt.show()
#Colocando os Dados na mesma escala

from sklearn.preprocessing import MinMaxScaler



#colsDFSel = ['num_gestacoes', 'glicose', 'pressao_sanguinea','insulina', 'bmi', 'indice_historico', 'grossura_pele', 'idade','classe']

colsDFSel = ['num_gestacoes', 'glicose', 'insulina', 'bmi', 'indice_historico', 'idade', 'classe']





#colsDFTesteSel = ['num_gestacoes', 'glicose', 'pressao_sanguinea','insulina', 'bmi', 'indice_historico', 'grossura_pele', 'idade']

colsDFTesteSel = ['num_gestacoes', 'glicose', 'insulina', 'bmi', 'indice_historico','idade']



colTrain = colsDFSel

dfMLTrain = df[colTrain]

arrayTrain = dfMLTrain.values



colTest =  colsDFTesteSel

dfMLTest = dfTest[colTest]

arrayTest = dfMLTest.values



#Fazendo split nos dados de treino

XTrain = arrayTrain[:,0:6]

YTrain = arrayTrain[:,6]

XTest = arrayTest[:,0:6]



#Criando escala

scaler = MinMaxScaler(feature_range = (0, 1))

rescaledXTrain = scaler.fit_transform(XTrain)

rescaledXTest = scaler.fit_transform(XTest)



#Dados na escala

print(rescaledXTrain[0:5,:])
#Fazendo Feature Selection usando chi2 test



#Bibliotecas

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



#Selecionando as 5 melhores features para usar no modelo

test = SelectKBest(score_func = chi2, k = 5) 

fit = test.fit(rescaledXTrain, YTrain)



print("====== Features =======\n", colsDFSel)



#Score

print(fit.scores_)

print("========================\n")

features = fit.transform(rescaledXTrain)



#Sumarizando Features selecionadas

print(features[0:5,:])
#Comparação de modelos

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV



#Definindo numero de folds

num_folds = 10

seed = 7





#Preparando modelos

modelos = []

modelos.append(('LR', LogisticRegression()))

modelos.append(('LDA', LinearDiscriminantAnalysis()))

modelos.append(('NB', GaussianNB()))

modelos.append(('KNN', KNeighborsClassifier()))

modelos.append(('DTC', DecisionTreeClassifier()))

modelos.append(('SVM', SVC()))

modelos.append(('ETC', ExtraTreesClassifier()))

modelos.append(('ABC', AdaBoostClassifier()))

modelos.append(('BGC', BaggingClassifier()))

modelos.append(('RFC', RandomForestClassifier()))

modelos.append(('GPC', GaussianProcessClassifier())) #===========> não deve aceitar valores negativos

modelos.append(('MLPC', MLPClassifier()))





#Avaliação dos modelos

resultados = []

nomes = []



for nome, modelo in modelos:

    kfold = model_selection.KFold(n_splits = num_folds, random_state = seed)

    cv_results = model_selection.cross_val_score(modelo, rescaledXTrain, YTrain, cv = kfold, scoring = 'accuracy')

    resultados.append(cv_results)

    nomes.append(nome)

    msg = "%s: %f (%f)" % (nome, cv_results.mean(), cv_results.std())

    print(msg)



# Boxplot pra comparar os modelos

fig = plt.figure()

fig.suptitle('Comparação dos Modelos')

ax = fig.add_subplot(111)

plt.boxplot(resultados)

ax.set_xticklabels(nomes)

plt.show()
#KNN

modelKNN = KNeighborsClassifier()



modelKNN.fit(rescaledXTrain, YTrain)



param_grid_knn = { 

    'n_neighbors': [3, 4, 5, 6, 7],

    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],

    'weights': ['uniform','distance'],

    'leaf_size':[1, 2, 3],

    'p': [2, 1]  

}

CV_knn = GridSearchCV(estimator=modelKNN, param_grid=param_grid_knn, cv=num_folds)

CV_knn.fit(rescaledXTrain, YTrain)

print (CV_knn.best_params_)

print (round(CV_knn.score(rescaledXTrain, YTrain) * 100, 2))





# Predictions

YPredKNN = CV_knn.predict(rescaledXTest)
#DTC

modelDTC = DecisionTreeClassifier()





param_grid_dtc = { 

    'min_samples_split': [0.1, 0.15, 0.2, 0.5],

    'min_samples_leaf': [0.1, 0.5, 1, 2, 3],

    'max_depth': [None, 1, 2],

    'criterion': ['gini','entropy'],

    'presort':[True,False]

}



CV_dtc = GridSearchCV(estimator=modelDTC, param_grid=param_grid_dtc, cv=num_folds)

CV_dtc.fit(rescaledXTrain, YTrain)

print (CV_dtc.best_params_)

print (round(CV_dtc.score(rescaledXTrain, YTrain) * 100, 2))



#Predictions

YPredDTC = CV_dtc.predict(rescaledXTest)
#RFC

modelRFC = RandomForestClassifier()



param_grid_rfc = { 

    'n_estimators': [1, 2, 3],

    'max_features': ['auto', 'sqrt', 'log2'],

    'criterion': ['gini','entropy'],

    'bootstrap':[True,False],

    'verbose':[0, 1],

    'warm_start':[True,False]

}



CV_rfc = GridSearchCV(estimator=modelRFC, param_grid=param_grid_rfc, cv=num_folds)

CV_rfc.fit(rescaledXTrain, YTrain)

print (CV_rfc.best_params_)

print (round(CV_rfc.score(rescaledXTrain, YTrain) * 100, 2))





# Predictions

YPredRFC = CV_rfc.predict(rescaledXTest)
#ABC

modelABC = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))

param_grid_abc = {

    'n_estimators': [200],

    'algorithm': ['SAMME', 'SAMME.R'],

    'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5, 1]

}

CV_abc = GridSearchCV(estimator=modelABC, param_grid=param_grid_abc, cv=num_folds)

CV_abc.fit(rescaledXTrain, YTrain)

print (CV_abc.best_params_)



print (round(CV_abc.score(rescaledXTrain, YTrain) * 100, 2))



# Predictions

YPredABC = CV_abc.predict(rescaledXTest)
#Gerando o resultado para o Submission File

resultado = YPredABC.astype(int)

#resultado = YPredRFC.astype(int)

#resultado = YPredDTC.astype(int)

#resultado = YPredKNN.astype(int)
#Criando Submission file

submission = pd.DataFrame({

        "id": dfTest['id'],

        "classe": resultado

    })



submission.to_csv('submission.csv', index=False)