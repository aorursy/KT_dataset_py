

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split

from sklearn.metrics import make_scorer, mean_absolute_error, r2_score

import xgboost



%matplotlib inline

warnings.filterwarnings("ignore")
df_train = pd.read_csv('../input/train_V2.csv', nrows=2000)

df_test  = pd.read_csv('../input/test_V2.csv', nrows=2000)
df_train.describe()
df_test.describe()
df_train = df_train.drop(df_train[df_train.winPlacePerc.isnull()].index,inplace = False)
f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(df_train.corr(), annot=True, linewidths=.5, fmt='.1f',ax=ax)

plt.show()
k = 5

f,ax = plt.subplots(figsize=(6, 6))

cm = df_train.corr().nlargest(k, 'winPlacePerc')

cols = cm.index

cm = cm[cols]

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
def obter_features(df):

    #Obter a quantidade de jogadores por partida

    df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')



    #Obter taxa de mortes por jogador por partida

    df['killsPerMatch'] = df['kills'] / df['playersJoined']

    df['killsPerMatch'].fillna(0,inplace=True)



    #Obter taxa de dano por jogador por partida

    df['damagePerMatch'] = df['damageDealt'] / df['playersJoined']

    df['damagePerMatch'].fillna(0,inplace=True)



    #Obter quantidade média de dano por morte

    df['damagePerKill'] = df['damageDealt'] / df['kills']

    df['damagePerKill'].fillna(0,inplace=True)

    df['damagePerKill'].replace(np.inf,0,inplace=True)



    #Obter taxa de tiros na cabeça por morte

    df['headshotPerKill'] = df['headshotKills'] / df['kills']

    df['headshotPerKill'].fillna(0, inplace=True) 



    #Obter distância total percorrida pelo jogador na partida

    df['totalDistance'] = df['rideDistance'] + df['swimDistance'] + df['walkDistance']

    

    return df
df_train = obter_features(df_train)

df_test = obter_features(df_test)
features = df_train.columns

features = features.drop(['Id', 'groupId', 'matchId', 'winPlacePerc', 'matchType'])

features
f,ax = plt.subplots(figsize=(8, 8))

new_features = df_train[['playersJoined', 'killsPerMatch', 'damagePerMatch', 'damagePerKill', 'headshotPerKill', 'totalDistance', 'winPlacePerc']]

sns.heatmap(new_features.corr(), annot=True, linewidths=.5, fmt='.1f',ax=ax)

plt.show()
#Separando a Classe das demais variáveis

target = df_train['winPlacePerc']

ids_train = df_train['Id']

ids_test = df_test['Id']

#Retirando também as variáveis winPlacePerc (alvo), Id, groupId e matchId

train_norm = np.array(df_train.drop(['Id', 'groupId', 'matchId', 'winPlacePerc', 'matchType'], axis=1))

test_norm = np.array(df_test.drop(['Id', 'groupId', 'matchId', 'matchType'], axis=1))
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder



# labelencoder_train = LabelEncoder()

# train_norm[:,12] = labelencoder_train.fit_transform(train_norm[:,12])

# onehotencoder = OneHotEncoder(categorical_features=[12])

# train_norm = onehotencoder.fit_transform(train_norm).toarray()



# labelencoder_test = LabelEncoder()

# test_norm[:,12] = labelencoder_test.fit_transform(test_norm[:,12])

# onehotencoder = OneHotEncoder(categorical_features=[12])

# test_norm = onehotencoder.fit_transform(test_norm).toarray()
# #Normlizando usando o StandardScaler

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()



# train_norm = scaler.fit_transform(train_norm)

# #pd.DataFrame(train_norm).head()



# test_norm = scaler.fit_transform(test_norm)

train_norm = (train_norm-train_norm.min())/(train_norm.max()-train_norm.min())

test_norm = (test_norm-test_norm.min())/(test_norm.max()-test_norm.min())
train_norm.shape
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() 

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() 

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df



train_norm = reduce_mem_usage(pd.DataFrame(train_norm))

target = reduce_mem_usage(pd.DataFrame(target))



test_norm = reduce_mem_usage(pd.DataFrame(test_norm))
#Salvar os Ids de cada instância para ter como associar depois do split

train_norm = train_norm.join(ids_train)



del ids_train



X_train, X_test, Y_train, Y_test = train_test_split(train_norm, target, test_size=1/3, random_state=0)

#pd.DataFrame(X_train).describe()

#Separar Ids dos conjuntos

ids_X_train = X_train['Id']

ids_X_test = X_test['Id']



X_train = X_train.drop(['Id'], axis=1)

X_test = X_test.drop(['Id'], axis=1)
#Treinando o modelo

model = xgboost.XGBRegressor(max_depth=17, gamma=0.3, learning_rate= 0.1)

model.fit(X_train,Y_train)
xgboost.plot_importance(model)
pred = model.predict(test_norm)
submit_xg = pd.DataFrame({'Id': ids_test, "winPlacePerc": pred} , columns=['Id', 'winPlacePerc'])



# r2_test_XGB = r2_score(Y_test,pred)

# mae_test_XGB = mean_absolute_error(Y_test,pred)



# print('XGBoost Resultados para o conjunto de testes:')

# print('Índice R^2: ' + str(r2_test_XGB))

# print('Erro médio absoluto: ' + str(mae_test_XGB))

print(submit_xg.head())

#submit_xg.to_csv("submission.csv", index = False)
from sklearn.tree import DecisionTreeRegressor 

regressor = DecisionTreeRegressor()

regressor.fit(X_train,Y_train) #X são os previsores e Y os valores correspondentes

#Para fazer uma previsão:

previsoes = regressor.predict(X_test)
score_train_DT = regressor.score(X_train,Y_train) #Valor do score na base de dados de treinamento

score_test_DT = regressor.score(X_test,Y_test) #Valor do Score na base de dados de teste

acuracia_DT = r2_score(Y_test, previsoes)

print (score_train_DT)

print (score_test_DT)

print (acuracia_DT)

#calculando o erro de uma árvore de decisão para regressão:

mae_DT = mean_absolute_error(Y_test,previsoes)

#mae contém o valor do mean absolute error

print (mae_DT)
#f,ax = plt.subplots(figsize=(20, 20))

#sns.heatmap(df_train.corr(), annot=True, linewidths=.5, fmt='.1f',ax=ax)

#plt.show()

def performance_metric(y_true, y_predict):

    score = r2_score(y_true,y_predict)

    return score
def grid_scores_to_df(grid_scores):

    """

    Convert a sklearn.grid_search.GridSearchCV.grid_scores_ attribute to a tidy

    pandas DataFrame where each row is a hyperparameter-fold combinatination.

    """

    rows = list()

    for grid_score in grid_scores:

        for fold, score in enumerate(grid_score.cv_validation_scores):

            row = grid_score.parameters.copy()

            row['fold'] = fold

            row['score'] = score

            rows.append(row)

    df = pd.DataFrame(rows)

    return df
# Gerar conjuntos de validação-cruzada para o treinamento de dados

cv_sets = ShuffleSplit(n_splits=5, test_size = 0.25)



#n_estimators =10

rfr =  RandomForestRegressor(n_estimators=10, random_state=42)



#Gerar um dicionário para o parâmetro 'max_depth' com um alcance de 1 a 10

params ={'max_depth': list(range(1,5))}



#Transformar 'performance_metric' em uma função de pontuação utilizando 'make_scorer' 

scoring_fnc = make_scorer(performance_metric)



# Gerar o objeto de busca em matriz

grid = GridSearchCV(rfr, params, scoring=scoring_fnc, cv=cv_sets)



# Ajustar o objeto de busca em matriz com os dados para calcular o modelo ótimo

grid = grid.fit(X_train, Y_train)
# Usando o melhor modelo para predição

rfr = grid.best_estimator_

previsoes = rfr.predict(X_test)
#Valor do score na base de dados de treinamento

score_train_RFR = rfr.score(X_train,Y_train)



#Valor do Score na base de dados de teste

score_test_RFR = rfr.score(X_test,Y_test)

print ('Random Forest Regressor Results: ')

print ('Score de treino: ' + str(score_train_RFR))

print ('Score de teste: ' + str(score_test_RFR))



#calculando o erro de uma árvore de decisão para regressão:

mae_RFR = mean_absolute_error(Y_test,previsoes)

#mae contém o valor do mean absolute error

print ('Erro médio absoluto: ' + str(mae_RFR))



#Acurácia do modelo

r2_RFR = r2_score(Y_test, previsoes)

print ('Índice R²: ' + str(r2_RFR))
rfr_scores = grid_scores_to_df(grid.grid_scores_)

rfr_scores
from sklearn.svm import SVR



# Gerar conjuntos de validação-cruzada para o treinamento de dados

cv_sets = ShuffleSplit(n_splits=5, test_size = 0.25)



svr = SVR()



#Gerar um dicionário para o parâmetro 'max_depth' com um alcance de 1 a 10

params = {'kernel': ('rbf','linear','poly')}



#Transformar 'performance_metric' em uma função de pontuação utilizando 'make_scorer' 

scoring_fnc = make_scorer(performance_metric)



# Gerar o objeto de busca em matriz

grid = GridSearchCV(svr, params, scoring=scoring_fnc, cv=cv_sets)



# Ajustar o objeto de busca em matriz com os dados para calcular o modelo ótimo

grid = grid.fit(X_train, Y_train)
# Usando o melhor modelo para predição

svr = grid.best_estimator_

previsoes = svr.predict(X_test)
#Valor do score na base de dados de treinamento

score_train_SVR = svr.score(X_train,Y_train)



#Valor do Score na base de dados de teste

score_test_SVR = svr.score(X_test,Y_test)

print ('SVR Results: ')

print ('Score de treino: ' + str(score_train_SVR))

print ('Score de teste: ' + str(score_test_SVR))



#calculando o erro de uma árvore de decisão para regressão:

mae_SVR = mean_absolute_error(Y_test,previsoes)

#mae contém o valor do mean absolute error

print ('Erro médio absoluto: ' + str(mae_SVR))



#Acurácia do modelo

r2_SVR = r2_score(Y_test, previsoes)

print ('Índice R²: ' + str(r2_SVR))
svr_scores = grid_scores_to_df(grid.grid_scores_)

svr_scores