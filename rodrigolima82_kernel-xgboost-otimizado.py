# Importar os principais pacotes

import numpy as np

import pandas as pd

import itertools

import seaborn as sns

sns.set()



import matplotlib.pyplot as plt

%matplotlib inline



import time

import datetime

import gc



# Evitar que aparece os warnings

import warnings

warnings.filterwarnings("ignore")



# Seta algumas opções no Jupyter para exibição dos datasets

pd.set_option('display.max_columns', 200)

pd.set_option('display.max_rows', 200)



# Variavel para controlar o treinamento no Kaggle

TRAIN_OFFLINE = False
# Importa os pacotes de algoritmos

import xgboost as xgb

from xgboost.sklearn import XGBClassifier



# Importa pacotes do sklearn

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, log_loss

from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
def read_data():

    

    if TRAIN_OFFLINE:

        print('Carregando arquivo dataset_treino.csv....')

        train = pd.read_csv('../dataset/dataset_treino.csv')

        print('dataset_treino.csv tem {} linhas and {} colunas'.format(train.shape[0], train.shape[1]))

        

        print('Carregando arquivo dataset_teste.csv....')

        test = pd.read_csv('../dataset/dataset_teste.csv')

        print('dataset_teste.csv tem {} linhas and {} colunas'.format(test.shape[0], test.shape[1]))

        

    else:

        print('Carregando arquivo dataset_treino.csv....')

        train = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv')

        print('dataset_treino.csv tem {} linhas and {} colunas'.format(train.shape[0], train.shape[1]))

        

        print('Carregando arquivo dataset_treino.csv....')

        test = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv')

        print('dataset_teste.csv tem {} linhas and {} colunas'.format(test.shape[0], test.shape[1]))

    

    return train, test
# Leitura dos dados

train, test = read_data()
# Removendo todas as variaveis categoricas

drop_features = []

for col in train.columns:

    if train[col].dtype =='object':

        drop_features.append(col)



train = train.drop(drop_features, axis=1)
# Preenche os dados missing com 0 (zero)

train.fillna(train.mean(),inplace=True)
# Separando features preditoras e target

train_x = train.drop(['ID','target'], axis=1)

train_y = train['target']



# Padronizando os dados

scaler = StandardScaler()

train_x = scaler.fit_transform(train_x)
# Criando uma funcao para criação, execução e validação do modelo

def run_model(modelo, X_tr, y_tr, useTrainCV=True, cv_folds=5, early_stopping_rounds=10):

    

    # Utilização do Cross-Validation

    if useTrainCV:

        xgb_param = modelo.get_xgb_params()

        xgtrain = xgb.DMatrix(X_tr, label=y_tr)

        

        print ('Start cross validation')

        cvresult = xgb.cv(xgb_param, 

                          xgtrain, 

                          num_boost_round=modelo.get_params()['n_estimators'], 

                          nfold=cv_folds,

                          metrics=['logloss'],

                          stratified=True,

                          seed=42,

                          #verbose_eval=True,

                          early_stopping_rounds=early_stopping_rounds)



        modelo.set_params(n_estimators=cvresult.shape[0])

        best_tree = cvresult.shape[0]

        print('Best number of trees = {}'.format(best_tree))

    

    # Fit do modelo

    modelo.fit(X_tr, y_tr, eval_metric='logloss')

        

    # Predição no dataset de treino

    train_pred = modelo.predict(X_tr)

    train_pred_prob = modelo.predict_proba(X_tr)[:,1]

    

    # Exibir o relatorio do modelo

    #print("Acurácia : %.4g" % accuracy_score(y_tr, train_pred))

    #print("AUC Score (Treino): %f" % roc_auc_score(y_tr, train_pred_prob))

    print("Log Loss (Treino): %f" % log_loss(y_tr, train_pred_prob))

    print("Log Loss (Test): %f" % cvresult['test-logloss-mean'][best_tree-1])

    

    feature_imp = pd.Series(modelo.feature_importances_.astype(float)).sort_values(ascending=False)

    

    plt.figure(figsize=(18,8))

    feature_imp[:25].plot(kind='bar', title='Feature Importances')

    plt.ylabel('Feature Importance Score')

    plt.tight_layout()
%%time



# Criando o primeiro modelo XGB

modeloXGB = XGBClassifier(learning_rate = 0.1,

                          n_estimators = 200,

                          max_depth = 5,

                          min_child_weight = 1,

                          gamma = 0,

                          subsample = 0.8,

                          colsample_bytree = 0.8,

                          objective = 'binary:logistic',

                          n_jobs = -1,

                          scale_pos_weight = 1,

                          seed = 42)



run_model(modeloXGB, train_x, train_y)
gc.collect()
'''%%time



# Definindo os parametros que serão testados no GridSearch

param_v1 = {

 'max_depth':range(2,5),

 'min_child_weight':range(1,2)

}



grid_1 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 

                                                n_estimators = 200, 

                                                max_depth = 5,

                                                min_child_weight = 1, 

                                                gamma = 0, 

                                                subsample = 0.8, 

                                                colsample_bytree = 0.8,

                                                objective = 'binary:logistic', 

                                                nthread = 4, 

                                                scale_pos_weight = 1, 

                                                seed = 42),

                      param_grid = param_v1, 

                      scoring = 'neg_log_loss',

                      n_jobs = -1,

                      iid = False, 

                      cv = 5)



# Realizando o fit e obtendo os melhores parametros do grid

grid_1.fit(train_x, train_y)

grid_1.best_params_, grid_1.best_score_'''
'''%%time



# Definindo os parametros que serão testados no GridSearch

param_v2 = {

 'gamma':[i/10.0 for i in range(0,2)]

}



grid_2 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 

                                                n_estimators = 200, 

                                                max_depth = grid_1.best_params_['max_depth'],

                                                min_child_weight = grid_1.best_params_['min_child_weight'], 

                                                gamma = 0, 

                                                subsample = 0.8, 

                                                colsample_bytree = 0.8,

                                                objective = 'binary:logistic', 

                                                nthread = 4, 

                                                scale_pos_weight = 1, 

                                                seed = 42),

                      param_grid = param_v2, 

                      scoring = 'neg_log_loss',

                      n_jobs = -1,

                      iid = False, 

                      cv = 5)



# Realizando o fit e obtendo os melhores parametros do grid

grid_2.fit(train_x, train_y)

grid_2.best_params_, grid_2.best_score_'''
'''%%time



# Definindo os parametros que serão testados no GridSearch

param_v3 = {

 'subsample':[i/10.0 for i in range(6,8)],

 'colsample_bytree':[i/10.0 for i in range(6,8)]

}



grid_3 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 

                                                n_estimators = 200, 

                                                max_depth = grid_1.best_params_['max_depth'],

                                                min_child_weight = grid_1.best_params_['min_child_weight'], 

                                                gamma = grid_2.best_params_['gamma'], 

                                                subsample = 0.8, 

                                                colsample_bytree = 0.8,

                                                objective = 'binary:logistic', 

                                                nthread = 4, 

                                                scale_pos_weight = 1, 

                                                seed = 42),

                      param_grid = param_v3, 

                      scoring = 'neg_log_loss',

                      n_jobs = -1,

                      iid = False, 

                      cv = 5)



grid_3.fit(train_x, train_y)

grid_3.best_params_, grid_3.best_score_'''
'''%%time



# Definindo os parametros que serão testados no GridSearch

param_v4 = {

 'reg_alpha':[0, 0.001, 0.005]

}



grid_4 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 

                                                n_estimators = 200, 

                                                max_depth = grid_1.best_params_['max_depth'],

                                                min_child_weight = grid_1.best_params_['min_child_weight'], 

                                                gamma = grid_2.best_params_['gamma'], 

                                                subsample = grid_3.best_params_['subsample'], 

                                                colsample_bytree = grid_3.best_params_['colsample_bytree'],

                                                objective = 'binary:logistic', 

                                                nthread = 4, 

                                                scale_pos_weight = 1, 

                                                seed = 42),

                      param_grid = param_v4, 

                      scoring = 'neg_log_loss',

                      n_jobs = -1,

                      iid = False, 

                      cv = 5)



# Realizando o fit e obtendo os melhores parametros do grid

grid_4.fit(train_x, train_y)

grid_4.best_params_, grid_4.best_score_'''
'''%%time



# Criando o modelo XGB com todas as otimizações

modeloXGB_v2 = XGBClassifier(learning_rate = 0.01, 

                             n_estimators = 1000, 

                             max_depth = 4,

                             min_child_weight = 1,

                             gamma = 0.04, 

                             subsample = 0.6,

                             colsample_bytree = 0.8,

                             reg_alpha = 0, 

                             objective = 'binary:logistic', 

                             n_jobs = -1,

                             scale_pos_weight = 1, 

                             seed = 42)



run_model(modeloXGB_v2, train_x, train_y)'''
# Visualizando o modelo XGBoost

print(modeloXGB)
# Colocando o dataset de teste conforme o modelo treinado

# Neste caso é necessário aplicar a Feature Engineering usada para gerar o modelo

text_x = test.drop(['ID'], axis=1)



# Removendo todas as variaveis categoricas

drop_features = []

for col in text_x.columns:

    if text_x[col].dtype =='object':

        drop_features.append(col)

text_x = text_x.drop(drop_features, axis=1)



# Preenche os dados missing com 0 (zero)

text_x.fillna(text_x.mean(),inplace=True)



# Aplicando escala aos dados

text_x = scaler.fit_transform(text_x)



# Realizando as previsoes

test_pred_prob = modeloXGB.predict_proba(text_x)[:,1]
# Criando dataset de submissao

submission = pd.DataFrame({'ID': test["ID"], 'PredictedProb': test_pred_prob.reshape((test_pred_prob.shape[0]))})

print(submission.head(10))
submission.to_csv('submission.csv', index=False)
plt.hist(submission.PredictedProb)

plt.show()