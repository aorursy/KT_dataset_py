import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import importlib



# Suprimindo warnings para manter mais limpo os outputs

import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore", category=DeprecationWarning) 
# data frame de treino (leitura do arquivo de treino)

df = pd.read_csv('../input/abalone-train.csv',sep=',', index_col='id')
# vendo o formato (tamanho e colunas) do dataframe

df.shape
# visuzaliando as primeiras 5 linhas do datframe

df.head(5)

# utilizando o método describe para gerar dados estatísticos sumarizados do dataframe

df.describe()



#aparentemente todas as colunas têm valor já que o count é igual para todas elas
# Verificamando os tipos de dados 

df.dtypes



# somente sex é do tipo object. Necessário transformá-lo em valor numérico
#Transoformando a coluna sex em valores numéricos



#Substitituindo letras por números

df= df.replace('F',0)

df = df.replace ('M', 1)

df= df.replace('I',2)



#conferindo os tipos do dataFrame

df.dtypes
#Visualização da distribulição por sexo

sns.countplot(df['sex'],label="Count")



# dataframe c uma distribuição razoavelmente regular 
#Histogram de distribuição por numero de aneis

sns.countplot(df['rings'], label="Count")
# visualização relação grafica par a par de atributos do dframe

sns.pairplot(df, hue='sex');



# possível verificar visualmente alguns outliers
# Conferindo se realmente não há volores nulos para algum atributo

missing_values = df.isnull().sum().sort_values(ascending = False)

percentage_missing_values = (missing_values/len(df))*100

pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing'])



# Correlacao das variaveis do dataframe com mapa de calor

import matplotlib.pyplot as plt



fig, ax = plt.subplots(figsize=(18,10)) 



# removendo a coluna rings antes de fazer a correlação 

frame = df.iloc[:,:-1]

frame = frame.iloc[:,1:]

corr = frame.corr()



sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 11}

          , linewidths=0.1, linecolor='white', ax=ax)
# Importando modelos

from sklearn import linear_model

from sklearn import svm



from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import  RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import  GradientBoostingRegressor

from sklearn.linear_model import  Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import SGDRegressor

from sklearn.svm import SVR

from sklearn.svm import LinearSVR

from sklearn.svm import NuSVR



# array para armazenar modelos que serão testado (performance)

models = []



models.append(('KNN-4', KNeighborsRegressor(n_neighbors = 4) ))

models.append(('KNN-3', KNeighborsRegressor(n_neighbors = 3) ))

models.append(('RandomForestRegressor', RandomForestRegressor() ))

models.append(('LinearRegression', LinearRegression() ))

models.append(('Lasso', Lasso() ))

models.append(('Elastice Net', ElasticNet() ))

models.append(('Rdige', Ridge() ))

models.append(('SGDRegressor', SGDRegressor() ))

models.append(('SVR', SVR() ))

models.append(('Linear SRV', LinearSVR() ))

models.append(('NU SVR', NuSVR() ))

# importando cross_val_score para testes de performance

from sklearn.model_selection import cross_val_score
# definir dados de entrada

X = df.drop(['rings'], axis=1) # tudo, exceto a coluna alvo

#X = dfSelectedColumns

Y = df['rings'] # apenas a coluna alvo



# visualizando os nomes dos atributos do dataframe

df.columns[:-1]

# importando biblioteca para Standardize

from sklearn.preprocessing import StandardScaler



#transformando os atributos de X (dataframe)

scaler = StandardScaler().fit(X)

arrayX = scaler.transform(X)



columns = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',

       'viscera_weight', 'shell_weight']



# gerando um novo X (dataframe) já transformado (scaler)

Xrescaled = pd.DataFrame(arrayX, columns=columns, index=X.index)
#treinando a performance dos modelos (models[]) com cross validation

names = []

scores = []

for name, model in models:

    ##model.fit(X_train, y_train)

    score = (cross_val_score(model, Xrescaled, Y, cv=10)) # 10 iterações

    names.append(name)

    scores.append(score)



scoresConsolidado = []

for sc in scores:

    scoresConsolidado.append(np.average(sc))



results = pd.DataFrame({'Model': names, 'Score': scoresConsolidado})

results = results.sort_values(by=['Score'], ascending=False) 



#vendo resultados

results



#AQUI UMA DÚVIDA:

## QUAL SCORE DEVO CONSIDERAR? tinha entendido q pra regressao os primeiros seriam os do final da lista.

## mas os que perfomaram melhor no kaggle foram o 3 primeiros (SVR, NU SVR,  Ridge)
# armazenar os modelos que seram testados para identificar os melhores parâmetros para tunning

topModels = []



# SVR

# 'gamma' : [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],

tunedParam  = {'kernel' : ['rbf', 'linear', 'poly'],

               'gamma' : [0.1, 0.2, 0.3],

               'C' : [1.0, 2.0, 3.0]}

topModels.append([models[8][0], models[8][1], tunedParam])





# Ridge

tunedParam  = {'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],

               'tol' : [1e-3, 1e-4]}

topModels.append([models[6][0], models[6][1], tunedParam])

# Para visualizar os parametros possíveis de um model específico

#Lasso().get_params()
# apos fazer uma lista "topModels[]" de modelos mais performáticos utilizar o gridSearch para identificar os 

# melhores parâmetros para ser utilizado por esses modelos com o dataframe de treino

# importando gridSearch

from sklearn.model_selection import GridSearchCV



grids = []



names = []

bestScores = []

bestParams = []

bestStimators = []



for name, model, tunedParam in topModels:

    grid = GridSearchCV(estimator = model, param_grid = tunedParam ,scoring= 'r2' ,cv = 10,  n_jobs = -1)

    #grid = GridSearchCV(model, pGrid, cv=10, scoring='accuracy')

    grid.fit(Xrescaled, Y)

    names.append(name)

    bestScores.append(grid.best_score_)

    bestParams.append(grid.best_params_)

    bestStimators.append(grid.best_estimator_)

    

dfResults = pd.DataFrame({'Model': names, 'Score': bestScores})    
dfResults

#  visualizar melhores parametros pro SVR e Rdige

bestParams
#Tentativa Algoritimo:  SVR

# {'C': 3.0, 'gamma': 0.1, 'kernel': 'rbf'}

topModel = SVR(gamma = 0.1, kernel = 'rbf', C = 3.0)



# Treinamento do modelo

topModel.fit(Xrescaled, Y)



#lendo o arquivo de test

predictDF = pd.read_csv('../input/abalone-test.csv',sep=',', index_col='id')



#Substitituindo letras 'n' e 'y' por 0 e 1

predictDF = predictDF.replace ('M', 1)

predictDF = predictDF.replace('F',0)

predictDF = predictDF.replace('I',2)



#fazendo standardize

scaler = StandardScaler().fit(predictDF)

arrayX = scaler.transform(predictDF)



columns = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',

       'viscera_weight', 'shell_weight']



#novo X transformado (StandardScaler())

Xp = pd.DataFrame(arrayX, columns=columns, index=predictDF.index)



#Realizando a previção com o arquivo de teste

yP = topModel.predict(Xp)



# gerar dados de envio (submissão)

submission = pd.DataFrame({'id': predictDF.index,'rings': yP})

submission.set_index('id', inplace=True)

submission.to_csv('./abalone-SVR.csv')

#submission.to_csv('./abalone-SVR.csv')
'''

#Algoritimo escolhido: Lasso

topModel = Lasso(max_iter=1000, normalize=False, tol=0.001)



# Treinamento do modelo

topModel.fit(Xrescaled, Y)



#lendo o arquivo de test para prever

predictDF = pd.read_csv('../input/abalone-test.csv',sep=',', index_col='id')



#Substitituindo letras 'n' e 'y' por 0 e 1

predictDF = predictDF.replace ('M', 1)

predictDF = predictDF.replace('F',0)

predictDF = predictDF.replace('I',2)



scaler = StandardScaler().fit(predictDF)

arrayX = scaler.transform(predictDF)



columns = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',

       'viscera_weight', 'shell_weight']



Xp = pd.DataFrame(arrayX, columns=columns, index=predictDF.index)



yP = topModel.predict(Xp)



# gerar dados de envio (submissão)

submission = pd.DataFrame({'id': predictDF.index,'rings': yP})

submission.set_index('id', inplace=True)

submission.to_csv('./abalone-Lasso.csv')



'''
'''

#Segundo Algoritimo escolhido: ElasticNet

topModel = ElasticNet(max_iter=1000, normalize=False, selection='random', tol=0.001)



# Treinamento do modelo

topModel.fit(Xrescaled, Y)



#lendo o arquivo de test para prever

predictDF = pd.read_csv('../input/abalone-test.csv',sep=',', index_col='id')



#Substitituindo letras 'n' e 'y' por 0 e 1

predictDF = predictDF.replace ('M', 1)

predictDF = predictDF.replace('F',0)

predictDF = predictDF.replace('I',2)



scaler = StandardScaler().fit(predictDF)

arrayX = scaler.transform(predictDF)



columns = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',

       'viscera_weight', 'shell_weight']



Xp = pd.DataFrame(arrayX, columns=columns, index=predictDF.index)



yP = topModel.predict(Xp)



# gerar dados de envio (submissão)

submission = pd.DataFrame({'id': predictDF.index,'rings': yP})

submission.set_index('id', inplace=True)

submission.to_csv('./abalone-ElasticNet.csv')

'''
'''

#terceiro Algoritimo escolhido: ElasticNet

topModel = KNeighborsRegressor(n_neighbors = 3, algorithm='auto')



# Treinamento do modelo

topModel.fit(Xrescaled, Y)



#lendo o arquivo de test para prever

predictDF = pd.read_csv('../input/abalone-test.csv',sep=',', index_col='id')



#Substitituindo letras 'n' e 'y' por 0 e 1

predictDF = predictDF.replace ('M', 1)

predictDF = predictDF.replace('F',0)

predictDF = predictDF.replace('I',2)



scaler = StandardScaler().fit(predictDF)

arrayX = scaler.transform(predictDF)



columns = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',

       'viscera_weight', 'shell_weight']



Xp = pd.DataFrame(arrayX, columns=columns, index=predictDF.index)



yP = topModel.predict(Xp)



# gerar dados de envio (submissão)

submission = pd.DataFrame({'id': predictDF.index,'rings': yP})

submission.set_index('id', inplace=True)

submission.to_csv('./abalone-KNN2.csv')

'''
'''

#Tentativa Algoritimo: NUm SVR

# {'gamma': 0.2, 'kernel': 'rbf', 'tol': 0.001}]

topModel = NuSVR(gamma = 0.2, kernel = 'rbf', tol = 0.001)



# Treinamento do modelo

topModel.fit(Xrescaled, Y)



#lendo o arquivo de test para prever

predictDF = pd.read_csv('../input/abalone-test.csv',sep=',', index_col='id')



#Substitituindo letras 'n' e 'y' por 0 e 1

predictDF = predictDF.replace ('M', 1)

predictDF = predictDF.replace('F',0)

predictDF = predictDF.replace('I',2)



scaler = StandardScaler().fit(predictDF)

arrayX = scaler.transform(predictDF)



columns = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',

       'viscera_weight', 'shell_weight']



Xp = pd.DataFrame(arrayX, columns=columns, index=predictDF.index)



yP = topModel.predict(Xp)



# gerar dados de envio (submissão)

submission = pd.DataFrame({'id': predictDF.index,'rings': yP})

submission.set_index('id', inplace=True)

submission.to_csv('./abalone-NuSVR.csv')

'''
'''

#Tentativa Algoritimo:  Ridge

# 'solver': 'saga', 'tol': 0.001}

topModel = Ridge(solver = 'saga', tol = 0.001)



# Treinamento do modelo

topModel.fit(Xrescaled, Y)



#lendo o arquivo de test para prever

predictDF = pd.read_csv('../input/abalone-test.csv',sep=',', index_col='id')



#Substitituindo letras 'n' e 'y' por 0 e 1

predictDF = predictDF.replace ('M', 1)

predictDF = predictDF.replace('F',0)

predictDF = predictDF.replace('I',2)



scaler = StandardScaler().fit(predictDF)

arrayX = scaler.transform(predictDF)



columns = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',

       'viscera_weight', 'shell_weight']



Xp = pd.DataFrame(arrayX, columns=columns, index=predictDF.index)



yP = topModel.predict(Xp)



# gerar dados de envio (submissão)

submission = pd.DataFrame({'id': predictDF.index,'rings': yP})

submission.set_index('id', inplace=True)

submission.to_csv('./abalone-Ridge.csv')

'''