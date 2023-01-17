import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.metrics import confusion_matrix

import itertools

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier, Pool
pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
df_treino = pd.read_csv('../input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv')

df_teste = pd.read_csv('../input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv')
df = df_treino

df_t = df_teste

target = df.target

df = df.drop(['target'], axis=1)
# ****************************** Preprocessing train data ******************************



df = df.drop(columns=['ID'])



#Drop columns with high correlation

df = df.drop(columns=['v8', 'v12', 'v13', 'v25', 'v32', 'v33', 'v34', 'v41', 'v43', 'v46', 'v49', 'v53', 

                     'v54', 'v55', 'v60', 'v63', 'v64', 'v65', 'v67', 'v73', 'v76', 'v77', 'v83', 'v86',

                     'v89', 'v95', 'v96', 'v97', 'v128'])





# Drop - Category colums

df = df.drop(columns=['v22', 'v30', 'v113', 'v125', 'v79', 'v3', 'v74'])



# Drop - high skewed

df = df.drop(columns=['v19'])



# Drop - zeros

df = df.drop(columns=['v129', 'v56', 'v38'])



# One-hot encoding (Se saiu melhor que inserir diretamente as variáveis categoricas direto no catBoost)

df = pd.get_dummies(data=df, columns=['v24', 'v31', 'v52', 'v66','v75', 'v91', 'v107', 'v110', 'v47', 'v71', 'v112'])



# Drop dummies colums without importance

df = df.drop(columns=['v31_B', 'v31_C', 'v52_A', 'v52_D', 'v52_E', 'v52_F', 'v52_G', 'v52_H', 

                      'v52_I', 'v52_J', 'v52_K', 'v52_L', 'v75_A', 'v75_B', 'v75_C', 'v91_B', 'v91_C', 

                      'v91_D', 'v91_E', 'v91_F', 'v91_G', 'v107_A', 'v107_B', 'v107_D', 'v107_E', 'v107_F', 

                      'v107_G', 'v71_K', 'v112_A', 'v112_B', 'v112_C', 'v112_D', 'v112_E', 'v112_G', 'v112_H', 

                      'v112_I', 

                      'v112_J', 'v112_L', 'v112_M', 'v112_N', 'v112_P', 'v112_Q', 'v112_R', 'v112_S', 'v112_V'])
# ****************************** Preprocessing test data ******************************



df_t = df_t.drop(columns=['ID'])



# Drop columns with high correlation

df_t = df_t.drop(columns=['v8', 'v12', 'v13', 'v25', 'v32', 'v33', 'v34', 'v41', 'v43', 'v46', 'v49', 'v53', 

                     'v54', 'v55', 'v60', 'v63', 'v64', 'v65', 'v67', 'v73', 'v76', 'v77', 'v83', 'v86',

                     'v89', 'v95', 'v96', 'v97', 'v128'])





# Drop - Category colums

df_t = df_t.drop(columns=['v22', 'v30', 'v113', 'v125', 'v79', 'v3', 'v74'])



# Drop - high skewed

df_t = df_t.drop(columns=['v19'])



# Drop - zeros

df_t = df_t.drop(columns=['v129', 'v56', 'v38'])



# One-hot encoding (Se saiu melhor que inserir diretamente as variáveis categoricas direto no catBoost)

df_t = pd.get_dummies(data=df_t, columns=['v24', 'v31', 'v52', 'v66','v75', 'v91', 'v107', 'v110', 

                                                'v47', 'v71', 'v112'])



#Drop colums without importance

df_t = df_t.drop(columns=['v31_B', 'v31_C', 'v52_A', 'v52_D', 'v52_E', 'v52_F', 'v52_G', 'v52_H', 

                      'v52_I', 'v52_J', 'v52_K', 'v52_L', 'v75_A', 'v75_B', 'v75_C', 'v91_B', 'v91_C', 

                      'v91_D', 'v91_E', 'v91_F', 'v91_G', 'v107_A', 'v107_B', 'v107_D', 'v107_E', 'v107_F', 

                      'v107_G', 'v112_A', 'v112_B', 'v112_C', 'v112_D', 'v112_E', 'v112_G', 'v112_H', 

                      'v112_I', 

                      'v112_J', 'v112_L', 'v112_M', 'v112_N', 'v112_P', 'v112_Q', 'v112_R', 'v112_S', 'v112_V'])

features = df

features_t = df_t
# Preenche missing values 



imp = IterativeImputer(max_iter=1, random_state=0)

features_filled = imp.fit(features)



features_filled = imp.transform(features)
# Normalizando os dados



scaler = StandardScaler()

X_filtered = scaler.fit_transform(features_filled)
# Aplica GridSearch para tentar encontrar os melhores parâmetros para o modelo



# model = CatBoostClassifier()



# grid = {'learning_rate': [0.01, 0.03, 0.1],

#         'depth': [4, 5, 6, 7, 8, 10],

#         'iterations':[500, 1000, 5000],

#         'od_type': ['Iter'],

#         'od_wait':[100],

#         'bagging_temperature':[0.1, 1.0],

#         'l2_leaf_reg': [1, 2, 5, 7]}



# grid_search_result = model.grid_search(grid,

#                                        X=X_filtered, 

#                                        y=target,

#                                        plot=True)
# Criação do modelo para teste rápido de alguns cenários



model = CatBoostClassifier(learning_rate=0.01,

                           eval_metric='Logloss',

                           depth=9,

                           random_seed=19,

                           bagging_temperature=0.1,

                           iterations=600,

                           od_type='Iter',

                           od_wait=100,

                           l2_leaf_reg=1)



model.fit(X_filtered, target, plot=True)
# Aplica o transform do modelo para dados faltantes



features_t_filled = imp.transform(features_t)
# Aplica o transform dos dados normalizados



X_test_filtered = scaler.transform(features_t_filled)
# Aplica o modelo nos dados de teste



test_pred_prob = model.predict_proba(X_test_filtered)[:,1]
# Prepara Datadrame para submissão



submission = pd.DataFrame({'ID': df_teste["ID"], 'PredictedProb': test_pred_prob.reshape((test_pred_prob.shape[0]))})

print(submission.head(15))
# Cria arquivo de submissão



submission.to_csv('submission.csv', index=False)
plt.hist(submission.PredictedProb)

plt.show()