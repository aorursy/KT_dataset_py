## Competição DSA de Machine Learning - Edição Junho/2019

# MARCIO DE LIMA



# As submissões para esta competição serão avaliadas pelo RMSE (Root Mean Squared Error).

# Nesta competição, você desenvolverá algoritmos para identificar e atender as oportunidades mais relevantes 

# para os indivíduos, revelando sinais de lealdade dos clientes. Sua contribuição melhorará a vida dos 

# clientes e ajudará a reduzir as campanhas indesejadas, a fim de criar uma e experiência mais personalizada 

# para cada cliente e consequentemente aumentar a satisfação e claro, as vendas.
# Importando as bibliotecas

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from IPython.core.pylabtools import figsize

import seaborn as sns

import warnings



%matplotlib inline

warnings.filterwarnings("ignore")
# Carregando os arquivos

df = pd.read_csv('../input/dataset_treino.csv')

df_teste = pd.read_csv('../input/dataset_teste.csv')



df.head(5)
# Mostrando as estruturas dos Datasets

df.info()
df_teste.info()
#Limpando espaços do texto caso existam

df['first_active_month'] = df['first_active_month'].str.strip()

df_teste['first_active_month'] = df_teste['first_active_month'].str.strip()

#Checando valores NA nos dados

df.isna().any()[lambda x: x]

df_teste.isna().any()[lambda x: x]
df_teste.dropna()

df_teste.isna().any()[lambda x: x]
# Ajustando colunas e limpeza dos dados nos DataSets



#Criando novas colunas

df['Month'] = df.first_active_month.apply(lambda dt: dt[5:7])

df['Year'] = df.first_active_month.apply(lambda dt: dt[:4])

df_teste['Month'] = df_teste.first_active_month.apply(lambda dt: '01' if isinstance(dt,float)  else dt[5:7])

df_teste['Year'] = df_teste.first_active_month.apply(lambda dt: '2000' if isinstance(dt,float)  else dt[:4])



#Ajustando a tipagem da coluna

df['Month'] = df['Month'].apply(pd.to_numeric, downcast='integer')

df['Year'] = df['Year'].apply(pd.to_numeric, downcast='integer')

df_teste['Month'] = df['Month'].apply(pd.to_numeric, downcast='integer')

df_teste['Year'] = df['Year'].apply(pd.to_numeric, downcast='integer')



df.dtypes
# Criando coluna = calculo das features

df['feature_4'] = (df['feature_1'] * df['feature_1'].mean()) + (df['feature_2']  * df['feature_2'].mean()) + (df['feature_3'] * df['feature_3'].mean())

df_teste['feature_4'] = (df_teste['feature_1'] * df_teste['feature_1'].mean()) + (df_teste['feature_2']  * df_teste['feature_2'].mean()) + (df_teste['feature_3'] * df_teste['feature_3'].mean())

df.head(5)
# Correlação com a Variavel TARGET

df[df.columns.drop('target')].corrwith(df.target)

# Dados estatisticos

df.describe()
#Construindo um gráfico de HEATMAP

f, ax = plt.subplots(figsize=(15, 12))

sns.heatmap(df.corr(),linewidths=.5, ax=ax)
#Gerando gráficos para analise das variaveis



#Histogramas

df.plot(kind = 'hist', subplots = True, layout = (7,3), sharex = False, figsize=(20,70))

plt.show()
df.plot(kind = 'density', subplots = True, layout = (7,3), sharex = False, figsize=(20,70))

plt.show()
df.plot(kind = 'box', subplots = True, layout = (7,3), sharex = False, sharey = False, figsize=(20,70))

plt.show()
#Funcoes utilitárias para medir a performance dos modelos

from sklearn.metrics import mean_squared_error

from math import sqrt



def rmspe(y_test, y_pred):



    mse = mean_squared_error(y_test, y_pred)

    rmspe = sqrt(mse)   

    

    return rmspe



# Treinamento e resultado do modelo - funcao generica

def treine_e_avalie(model, X, y, X_test, y_test):

    

    

    # Predicao

    model_pred = treino_e_predicao(model, X, y, X_test)

    #Performance

    model_rmspe = rmspe(y_test, model_pred)

    

    # Retorno da Performance do modelo

    return model_rmspe



def treino_e_predicao(model, X, y, X_test):

    

    # FIT

    model.fit(X, y)

    # Predicao

    return model.predict(X_test)
df.shape
#Gerando dados de Treino e de Teste para os modelos

from sklearn.model_selection import train_test_split



seed = 1313



array = df.values

X = array[:,6:9]

Y = df.target.values



X_treino, X_teste, y_treino, y_teste = train_test_split(X, Y, test_size = 0.30, random_state = seed)

X_treino
# Importando os modelos



from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor
# Modelo 1 - Regressao Linear Simples



lr = LinearRegression()

lr_rmspe = treine_e_avalie(lr, X_treino, y_treino, X_teste, y_teste)



print('Modelo 1 - Regressao Linear => RMSPE = %0.4f' % lr_rmspe)
# Modelo 2 - KNN

knn = KNeighborsRegressor(n_neighbors=5)

knn_rmspe = treine_e_avalie(knn, X_treino, y_treino, X_teste, y_teste)



print('Modelo 2 - KNN => RMSPE = %0.4f' % knn_rmspe)
# Modelo 3 - GradientBoostingRegressor

gradient_boosted = GradientBoostingRegressor(random_state=60)

gradient_boosted_rmspe = treine_e_avalie(gradient_boosted, X_treino, y_treino, X_teste, y_teste)



print('Modelo 3 - GradientBoostingRegressor = %0.4f' % gradient_boosted_rmspe)

# Otimizando o modelo 3 - Otimização de Hyperparâmetro

from sklearn.model_selection import RandomizedSearchCV



#Modelo para testar a otimização

gbr = GradientBoostingRegressor(random_state=13)

#Parametros da otimização

param_grid = {

        'n_estimators': [100, 200, 500],

        'max_features': ['auto', 'sqrt', 'log2', None],

        'max_depth': [2, 3, 5, 10, 15],

        'learning_rate': [0.1],

        'loss': ['ls', 'lad', 'huber'],

        'subsample': [1]

}



#Modelo para melhor scoring para o RMSE

modelo_otm = RandomizedSearchCV(estimator=gbr,

                               param_distributions=param_grid,

                               cv=2, n_iter=1, 

                               scoring = 'neg_mean_absolute_error',

                               n_jobs = -1, verbose = 1, 

                               return_train_score = True,

                               random_state=60)

#Treinando o modelo otimizado

modelo_otm.fit(X_treino, y_treino)

#Resultado do modelo otimizado

print('Melhores Params:')

print(modelo_otm.best_params_)

print('Melhor CV Score:')

print(-modelo_otm.best_score_)
# Modelo 4 - GradientBoostingRegressor Otimizado

# Melhores parameters

#gradient_boosted1_otm = GradientBoostingRegressor(max_depth=2, max_features='sqrt',

#                                                  n_estimators=500, loss='lad', random_state=60, 

#                                                  learning_rate=0.1, verbose=1, subsample=1)



gradient_boosted1_otm = GradientBoostingRegressor( max_depth=3, max_features='sqrt', 

                                                   n_estimators=100, 

                                                   criterion='mse',

                                                   learning_rate=0.05,

                                                   random_state=60)



modelo_pred_otm = treino_e_predicao(gradient_boosted1_otm, X_treino, y_treino, X_teste)

gradient_boosted_rmspe_otm = rmspe(y_teste, modelo_pred_otm)



print('Modelo 4 - GradientBoostingRegressor - Otimizado = %0.4f' % gradient_boosted_rmspe_otm)

#Gerando os dados para o Arquivo de submissao

array = df_teste.values

X_teste1 = array[:,5:9]

X_teste1
#Gerando Arquivo de Submissao

df_submission = pd.DataFrame()

df_submission['card_id'] = df_teste['card_id']



resultado_otm = gradient_boosted1_otm.predict(X_teste1)
resultado_otm
df_submission['target'] = resultado_otm
df_submission.head(10)
#Gravando Arquivo de Submissao

df_submission.to_csv('submission.csv', index=False)