# Arrays, matrizes, cálculos e dataframes

import numpy as np

import pandas as pd



# Importando a métrica

from sklearn.metrics import accuracy_score



# Divisão dataframe

from sklearn.model_selection import train_test_split



# Métodos de árvores

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier



# GridSearch e Cross-validation

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold



# Gráficos

import matplotlib.pyplot as plt



# Importo o arquivo

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        df = pd.read_csv(os.path.join(dirname, filename))
# Visualizo

df.head()
# Renomeio as colunas

df.columns = ['inadim', 'vl_emp', 'vl_dev_hip', 'vl_prop', 'mot_emp', 'prof', 

              'anos_emp', 'nr_rel_dep', 'lin_cred_inadim', 'meses_lin_ant', 'lin_cred_rec', 

              'lin_cred_atuais', 'tx_div_mes']        



print('Qtd inicial de linhas: ', df.shape)



# Removo as linhas que possuem menos de 5 colunas preenchidas

df = df[~(df.count(1) <= 5)]

print('Qtd final de linhas: ', df.shape)



# Seto 'Other' em prof (Profissão), quando estiver nula

df['prof'] = np.where(df['prof'].isna(), 'Other', df['prof'])



# Seto 0 em vl_dev_hip (valor devido de hipoteca), quando estiver nulo

df['vl_dev_hip'] = np.where(df['vl_dev_hip'].isna(), 0, df['vl_dev_hip'])



# Para os casos onde existe inadimplência e o valor da casa está nulo, seto 0, isto porque, possivelmente, a pessoa já tenha perdido o imóvel

df['vl_prop'] = np.where(df['vl_prop'].isna(), 0, df['vl_prop'])



# Se o motivo do empréstimo estiver nulo e o valor da hipoteca for zero, seto HomeImp, caso contrário, DebtCon

df['mot_emp'] = np.where((df['mot_emp'].isna()) & (df['vl_dev_hip'] == 0), 'HomeImp', df['mot_emp'])

df['mot_emp'] = np.where((df['mot_emp'].isna()) & (df['vl_dev_hip'] > 0), 'DebtCon', df['mot_emp'])



# Seto 0 em anos_emp quando for nulo

df['anos_emp'] = np.where(df['anos_emp'].isna(), 0, df['anos_emp'])



# Preencho a coluna tx_div_mes com a média da mesma coluna, isso porque ninguém tem 0% da renda comprometida, então a média é mais interessante

df['tx_div_mes'].fillna(df['tx_div_mes'].mean(), inplace=True)



# Demais colunas, preencho com 0

df.fillna(0, inplace=True)



# Transformo texto em número

for col in df.columns:

    if df[col].dtype == 'object':

        df[col + '_n'] = df[col].astype('category').cat.codes
# Visualizo como ficaram os dados

df.head()
# Visualizo o tipo dos dados

df.info()
# Separo as colunas que vou utilizar

feats = [c for c in df.columns if c not in ['inadim', 'mot_emp', 'prof']]

feats_wt = [c for c in df.columns if c not in ['mot_emp', 'prof']]



# Separo as bases de treino e teste

train, test = train_test_split(df[feats_wt], test_size=0.15, random_state=42)



# Verifico o tamanho dos dataframes de treino e teste

train.shape, test.shape
# Testo com RandomForest

rf = RandomForestClassifier(n_estimators=200, random_state=42)

rf.fit(train[feats], train['inadim'])

print('Porcentagem de acerto: ', accuracy_score(test['inadim'], rf.predict(test[feats])))



# Checo a importância das variáveis neste modelo

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
# Testo com GBM

gbm = GradientBoostingClassifier(n_estimators=200, random_state=42)

gbm.fit(train[feats], train['inadim'])

print('Porcentagem de acerto: ', accuracy_score(test['inadim'], gbm.predict(test[feats])))



# Checo a importância das variáveis neste modelo

pd.Series(gbm.feature_importances_, index=feats).sort_values().plot.barh()
# Testo com XGBoost

xgb = XGBClassifier(n_estimators=200, random_state=42)

xgb.fit(train[feats], train['inadim'])

print('Porcentagem de acerto: ', accuracy_score(test['inadim'], xgb.predict(test[feats])))



# Checo a importância das variáveis neste modelo

pd.Series(xgb.feature_importances_, index=feats).sort_values().plot.barh()
# seto os valores que quero testar

params = {

    'n_estimators': [200, 400],

    'max_depth': [None, 0, 2, 4],

    'min_samples_split': [1, 2, 4],

    'min_samples_leaf': [1, 2, 4]

    #'max_features': [None, 'auto', 'log2'] removi, senão nem roda

}



# executo antes para conseguir uma melhor combinação de valores

rf = RandomForestClassifier(n_estimators=200, random_state=42)



# validação cruzada que retorna dobras estratificadas

# as dobras são feitas preservando a porcentagem de amostras para cada classe

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 42)



# aplicação da grid

gsc = GridSearchCV(estimator=rf, param_grid=params, cv=skf.split(train[feats], train['inadim'].values), n_jobs=4)

gsc.fit(train[feats], train['inadim'])

print (gsc.best_params_)
# Utilizo os parâmetros sugeridos e testo

rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=1, max_depth=None, min_samples_split=4, random_state=42)

rf.fit(train[feats], train['inadim'])

print('Porcentagem de acerto: ', accuracy_score(test['inadim'], rf.predict(test[feats])))



# Checo a importância das variáveis

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
# Grid para usar com XGBoost

params = {

        'min_child_weight': [1, 3, 5, 8, 10],

        'gamma': [0, 0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [2, 4, 6, 8, 10],

        'learning_rate': [0.2, 0.4, 0.5, 0.6, 1.0],

        'max_delta_step': [0, 3, 5, 10],

        'base_score': [0, 0.3, 0.5, 0.7, 1.0],

        'n_estimators': [200, 400, 600]

        }



xgb = XGBClassifier(random_state=42)



# cross-validation

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 42)



random_search = RandomizedSearchCV(xgb, 

                                   param_distributions=params,

                                   n_iter=6,

                                   scoring='roc_auc', 

                                   n_jobs=4, 

                                   cv=skf.split(train[feats], train['inadim'].values), 

                                   verbose=3, 

                                   random_state=42)



random_search.fit(train[feats], train['inadim'].values)



print('\n Melhor resultado:')

print(random_search.best_estimator_)
# Utilizo o resultado do RandomSearch

xgb = XGBClassifier(base_score=0.7, booster=None, colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.6, gamma=1.5, gpu_id=-1,

              importance_type='gain', interaction_constraints=None,

              learning_rate=0.5, max_delta_step=10, max_depth=10,

              min_child_weight=5, monotone_constraints=None,

              n_estimators=400, n_jobs=0, num_parallel_tree=1,

              objective='binary:logistic', random_state=42, reg_alpha=0,

              reg_lambda=1, scale_pos_weight=1, subsample=1.0, tree_method=None,

              validate_parameters=False, verbosity=None)



xgb.fit(train[feats], train['inadim'])

print('Porcentagem de acerto: ', accuracy_score(test['inadim'], xgb.predict(test[feats])))



# Checo a importância das variáveis com os novos parâmetros

pd.Series(xgb.feature_importances_, index=feats).sort_values().plot.barh()
# Grid para usar com XGBoost

params = {

        'learning_rate': [0.2, 0.4, 0.5, 0.6, 1.0],

        'n_estimators': [200, 400, 600]

        }



xgb = XGBClassifier(random_state=42)



# cross-validation

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 42)



random_search = RandomizedSearchCV(xgb, 

                                   param_distributions=params,

                                   n_iter=6,

                                   scoring='roc_auc', 

                                   n_jobs=4, 

                                   cv=skf.split(train[feats], train['inadim'].values), 

                                   verbose=3, 

                                   random_state=42)



random_search.fit(train[feats], train['inadim'].values)



print('\n Melhor resultado:')

print(random_search.best_estimator_)
# Utilizo o resultado do RandomSearch

xgb = XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

              importance_type='gain', interaction_constraints=None,

              learning_rate=0.4, max_delta_step=0, max_depth=6,

              min_child_weight=1, monotone_constraints=None,

              n_estimators=600, n_jobs=0, num_parallel_tree=1,

              objective='binary:logistic', random_state=42, reg_alpha=0,

              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

              validate_parameters=False, verbosity=None)



xgb.fit(train[feats], train['inadim'])

print('Porcentagem de acerto: ', accuracy_score(test['inadim'], xgb.predict(test[feats])))



# Checo a importância das variáveis

pd.Series(xgb.feature_importances_, index=feats).sort_values().plot.barh()
# Apresento os valores referentes aos registros previstos e não previstos

test['t_inadim'] = xgb.predict(test[feats]).astype(int)

test['acerto'] = test['t_inadim'] == test['inadim']

test['acerto'] = np.where(test['acerto'] == True, 'Previsto', 'Não Previsto')

print('Previstos: ', str(np.round(test['acerto'].value_counts(normalize=True)[0], 3)) + '%', '-', test['acerto'].value_counts()[0])

print('Não Previstos: ', str(np.round(test['acerto'].value_counts(normalize=True)[1], 3)) + '%', '-', test['acerto'].value_counts()[1])
# Apresento o resultado em gráfico

pd.value_counts(test['acerto']).plot.bar(color='#4682B4')

plt.xticks(rotation=0)

plt.title('Total de registros previstos e não previstos')

plt.show()