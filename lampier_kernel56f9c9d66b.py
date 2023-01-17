# Para mostrar todas as saídas das células do Jupyter

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'
aleatorio = 7
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
df_treino = pd.read_csv('dataset_treino.csv', 

                        dtype={'feature_1': int, 'feature_2': int, 'feature_3': int})

df_treino.shape

df_treino.head()
df_teste = pd.read_csv('dataset_teste.csv',

                      dtype={'feature_1': int, 'feature_2': int, 'feature_3': int})

df_teste.shape

df_teste.head()
# Relação entre os datasets de teste e treino

df_teste.shape[0] / df_treino.shape[0]
df_sample_submission = pd.read_csv('sample_submission.csv')

df_sample_submission.shape

df_sample_submission.head()
# Garbage collector

import gc

gc.collect()
# Vamos utilizar o pandas_profiling para acelerar a análise exploratória inicial

import pandas_profiling as pp
# Conversão data coluna first_active_month tipo data e extração do mês e ano

df_treino['data_ativacao'] = pd.to_datetime(df_treino['first_active_month'])

df_treino['mes_ativacao'] = df_treino.data_ativacao.apply(lambda dt: dt.month)

df_treino['ano_ativacao'] = df_treino.data_ativacao.apply(lambda dt: dt.year)



# Ajustando tipo dado

# df_treino.mes_ativacao.astype('int', inplace=True)

# df_treino.ano_ativacao.astype('int', inplace=True)



# Remoção da coluna que não precisamos mais

df_treino.drop('first_active_month', axis=1, inplace=True)

df_treino.drop('data_ativacao', axis=1, inplace=True)



# Reorganizando as colunas

df_treino = df_treino[['card_id', 'mes_ativacao', 'ano_ativacao', 'feature_1', 'feature_2', 'feature_3', 'target']]



df_treino.head()
# Relatório automático do dataset

pp.ProfileReport(df_treino)
# Note um outlier antes de -30

# São 2207 registros (1,093% do total de registros) com valor de -33.21928095

df_treino['target'].hist(bins=500)
df_treino['ano_ativacao'].hist(bins=50)
# Conversão para data e extração do mês e ano de ativação

df_teste['data_ativacao'] = pd.to_datetime(df_teste['first_active_month'])

df_teste['mes_ativacao'] = df_teste.data_ativacao.apply(lambda dt: dt.month)

df_teste['ano_ativacao'] = df_teste.data_ativacao.apply(lambda dt: dt.year)



# Ajustando tipo dado

# df_treino.mes_ativacao.astype('int', inplace=True)

# df_treino.ano_ativacao.astype('int', inplace=True)



# Remoção da coluna que não precisamos mais

df_teste.drop('first_active_month', axis=1, inplace=True)

df_teste.drop('data_ativacao', axis=1, inplace=True)



# Reorganizando as colunas

df_teste = df_teste[['card_id', 'mes_ativacao', 'ano_ativacao', 'feature_1', 'feature_2', 'feature_3']]
# Relatório automático do dataset

pp.ProfileReport(df_teste)
df_treino.head()

df_teste.head()
# Não precisamos do campo card_id do arquivo de treino

df_treino.drop('card_id', axis=1, inplace=True)

df_treino.head()
import xgboost as xgb

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split

import numpy as np



# Separando as variáveis preditoras da variável predita

X, y = df_treino.iloc[:, :-1], df_treino.iloc[:, -1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 

                                                    shuffle=True, random_state=aleatorio)





gbm_param_grid = {

    'learning_rate': [0.01, 0.1, 1],

    'n_estimators': [500, 1000, 2000],

    'max_features': ['auto', 'sqrt', 'log2', None],

    'loss': ['huber'],

    'subsample': [1],

    'max_depth':[2, 3, 5],

    'nthread': [4]

}



gbm = xgb.XGBRegressor()



# n_jobs=-1 é usar todos os processadores

grid_mse = RandomizedSearchCV(estimator=gbm, param_distributions=gbm_param_grid, 

                        scoring='neg_mean_squared_error', cv=2, verbose=1, n_iter=5, 

                        return_train_score=True, random_state=aleatorio, n_jobs=1)



grid_mse.fit(X, y)



print('Melhores parâmetros encontrados no treio: ', grid_mse.best_params_)

print('Menor RMSE encontrado no treino: ', np.sqrt(np.abs(grid_mse.best_score_)))



# Predição

preds = grid_mse.predict(X_test)



print("RMSE nos dados de teste: ", np.sqrt(mean_squared_error(y_test, preds)))
predictions = grid_mse.predict(df_teste[['mes_ativacao', 'ano_ativacao', 'feature_1', 'feature_2', 'feature_3']])



submission_df = pd.DataFrame({"card_id":df_teste["card_id"].values})

submission_df["target"] = predictions

submission_df.to_csv("submission_2.csv", index=False)