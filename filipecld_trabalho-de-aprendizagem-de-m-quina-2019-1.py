import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

pd.set_option('display.max_columns', 100)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.info()
train.tail()
train.shape
# test head

test.head()
train.drop_duplicates()

train = train.drop(['id'], axis = 1)

train.shape
test.drop_duplicates()

test_id = test['id']

test = test.drop(['id'], axis = 1)

test.shape
test.head()
colunas = train.columns.tolist()

colunas_reg = [col for col in colunas if 'reg' in col]

colunas_cat = [col for col in colunas if 'cat' in col]

colunas_bin = [col for col in colunas if 'bin' in col]

colunas_car = [col for col in colunas if 'car' in col and 'cat' not in col]

colunas_calc = [col for col in colunas if 'calc' in col]

print(colunas_cat)
train.loc[:,colunas_reg].describe()
train.loc[:, colunas_car].describe()
id_0 = train[train.target == 0].index

id_1 = train[train.target == 1].index

proporção = id_1.shape[0]/train.shape[0]

print('Qual a probabilidade a priori de um segurado sinistrar: {}'. format(proporção))
var_missing = []



for f in train.columns:

    missings = train[train[f] == -1][f].count()

    if missings > 0:

        var_missing.append(f)

        missings_perc = missings/train.shape[0]

        

        print('Variável {} tem {} exemplos ({:.2%}) com valores omissos'.format(f, missings, missings_perc))

        

print('No total, existem {} variáveis com valores omissos'.format(len(var_missing)))
# Excluindo variáveis com muitos dados omissos

variaveis_excluir = ['ps_car_03_cat', 'ps_car_05_cat']

train.drop(variaveis_excluir, inplace=True, axis=1)

train.drop(colunas_calc, inplace = True, axis = 1)

# Imputando com a média e a moda

media_imp = Imputer(missing_values=-1, strategy='mean', axis=0)

moda_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)

train['ps_reg_03'] = media_imp.fit_transform(train[['ps_reg_03']]).ravel()

train['ps_car_14'] = media_imp.fit_transform(train[['ps_car_14']]).ravel()

train['ps_car_11'] = moda_imp.fit_transform(train[['ps_car_11']]).ravel()
# excluir a variável 'ps_car_03_cat' e 'ps_car_05_cat'

colunas_cat.remove('ps_car_03_cat')

colunas_cat.remove('ps_car_05_cat')

print(colunas_cat)
for i in colunas_cat:

    plt.figure()

    fig, ax = plt.subplots(figsize = (20,10))

    sns.barplot(ax = ax, x = i, y = 'target', data = train)

    plt.ylabel('% target', fontsize = 18)

    plt.xlabel(i, fontsize = 18)

    plt.tick_params(axis = 'both', which= 'major', labelsize = 18)

    plt.show()
continuas = [colunas_reg, colunas_car]

def correl(t):

    correlacao = train[t].corr()

    cmap = sns.diverging_palette(220, 10, as_cmap = True)



    fig, ax = plt.subplots(figsize = (10,10))

    sns.heatmap(correlacao, cmap = cmap, vmax = 1.0, center = 0, fmt = '.2f',

           square = True, linewidths = .5, annot = True, cbar_kws ={"shrink": .75})

    plt.show();

    

# Variáveis reg

for j in continuas:

    print('Heat Map de correlações para as variáveis {}' .format(j))

    correl(j)
for i in train.columns:

    if train[i].dtype == 'int64' and i != 'target':

        train[i] = train[i].astype('category')
# Checando

train.info()
# função get_dummies transforma as categorias em variáveis binárias

train = pd.get_dummies(train)

train.head()
# checando a dimensão

train.shape
X = train.drop(["target"], axis = 1)

y = train["target"]
X_train, X_test, y_train, y_test = train_test_split(

    X, y, stratify=y, random_state=0)
scaler = MinMaxScaler()

scaler.fit(X)

X = scaler.transform(X)
def gini(actual, pred):

    assert (len(actual) == len(pred))

    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)

    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]

    totalLosses = all[:, 0].sum()

    giniSum = all[:, 0].cumsum().sum() / totalLosses



    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)





def gini_normalized(actual, pred):

    return gini(actual, pred) / gini(actual, actual)
# Testando com a regressão logística com penalização L2

lr = LogisticRegression(penalty='l2', random_state=1)

lr.fit(X_train, y_train)

prob = lr.predict_proba(X_test)[:,1]

print("Índice de Gini normalizado para a Regressão Logística: ",gini_normalized(y_test, prob))
# Random Forest com 20 árvores

rf = RandomForestClassifier(n_estimators = 20, max_depth = 4, random_state = 1, max_features = 20)

rf.fit(X_train, y_train)

predictions_prob = rf.predict_proba(X_test)[:,1]

print("Índice de Gini normalizado para o Random Forest: ", gini_normalized(y_test, predictions_prob))
# taxa de aprendizagem = 0.05

xgbm = XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.05, random_state = 1)

xgbm.fit(X_train, y_train)

prob_xgb = xgbm.predict_proba(X_test)[:,1]

print("--------------------------------------------------------------------------------------------")

print("Índice de Gini normalizado para o XGBoost com learning_rate = 0.05: ", gini_normalized(y_test, prob_xgb))

print("--------------------------------------------------------------------------------------------")
# Optamos por tentar submeter o modelo com melhor score nos dados X_test - XGBoost com . 

# Testaremos nos dados completo de treinamento.

prob_xgb_y = xgbm.predict_proba(X)[:,1]

print("--------------------------------------------------------------------------------------------")

print("Índice de Gini normalizado para o XGBoost com learning_rate = 0.05 dados treino: ", gini_normalized(y, prob_xgb_y))

print("--------------------------------------------------------------------------------------------")
# Excluindo variáveis com muitos dados omissos

variaveis_excluir = ['ps_car_03_cat', 'ps_car_05_cat']

test.drop(variaveis_excluir, inplace=True, axis=1)

test.drop(colunas_calc, inplace = True, axis = 1)

# Imputando com a média e a moda

media_imp = Imputer(missing_values=-1, strategy='mean', axis=0)

moda_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)

test['ps_reg_03'] = media_imp.fit_transform(test[['ps_reg_03']]).ravel()

test['ps_car_14'] = media_imp.fit_transform(test[['ps_car_14']]).ravel()

test['ps_car_11'] = moda_imp.fit_transform(test[['ps_car_11']]).ravel()



# categorizando 

for i in test.columns:

    if test[i].dtype == 'int64' and i != 'target':

        test[i] = test[i].astype('category')

        

# função get_dummies transforma as categorias em variáveis binárias

test = pd.get_dummies(test)
# Normalização

scaler = MinMaxScaler()

scaler.fit(test)

test = scaler.transform(test)
# Aplicando no modelo XGBoost

y_test = xgbm.predict_proba(test)[:,1]



# Em results_df está a base de teste escorada, a coluna target possui as probabilidades

results_df = pd.DataFrame(data={'id':test_id, 'target':y_test})

print(results_df)

results_df.to_csv('submissão_1.csv', index=False)
# Testando nos dados sem exclusão de variáveis

#train = pd.read_csv('../input/train.csv')

#X = train.iloc[:,2:]

#y = train.iloc[:,1:2]

#X = train.drop(["id","target"], axis = 1)

#y = train["target"]

# Separando um conjunto para avaliar

#X_train, X_test, y_train, y_test = train_test_split(

#    X, y, stratify=y, random_state=0)

# Normalização Min-Max

#scaler = MinMaxScaler()

#scaler.fit(X_train)

#X_train = scaler.transform(X_train)

#scaler = MinMaxScaler()

#scaler.fit(X_test)

#X_test = scaler.transform(X_test)

# Modelos

# Regressão Logística

# Testando com a regressão logística com penalização L2

#lr = LogisticRegression(penalty='l2', random_state=1)

#lr.fit(X_train, y_train)

#prob_lr = lr.predict_proba(X_test)[:,1]

#print("--------------------------------------------------------------------------------------------")

#print("Índice de Gini normalizado para a Regressão Logística: ",gini_normalized(y_test, prob_lr))

#print("--------------------------------------------------------------------------------------------")

# Random Forest com 200 árvores

#rf = RandomForestClassifier(n_estimators = 200, max_depth = 4, random_state = 1, max_features = 15)

#rf.fit(X_train, y_train)

#prob_rf = rf.predict_proba(X_test)[:,1]

#print("--------------------------------------------------------------------------------------------")

#print("Índice de Gini normalizado para o Random Forest: ", gini_normalized(y_test, prob_rf))

#print("--------------------------------------------------------------------------------------------")

# XGBoost 

#xgbm = XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.05, random_state = 1)

#xgbm.fit(X_train, y_train)

#prob_xgb = xgbm.predict_proba(X_test)[:,1]

#print("--------------------------------------------------------------------------------------------")

#print("Índice de Gini normalizado para o XGBoost: ", gini_normalized(y_test, prob_xgb))

#print("--------------------------------------------------------------------------------------------")

# LightGBM

#lgb = LGBMClassifier(n_estimators = 100, learning_rate = 0.02, subsample = 0.7, num_leaves = 15, seed = 1)

#lgb.fit(X_train, y_train)

#prob_lgb = lgb.predict_proba(X_test)[:, 1]

#print("--------------------------------------------------------------------------------------------")

#print("Índice de Gini normalizado para o LightGBM: ", gini_normalized(y_test, prob_lgb))

#print("--------------------------------------------------------------------------------------------")



import pandas as pd

from pandas import read_csv, DataFrame

import numpy as np

from numpy.random import seed

from sklearn.preprocessing import minmax_scale

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn import datasets

from keras.layers import Input, Dense

from keras.models import Model

from matplotlib import pyplot as plt
# Carregamento das bases de treinamento e teste em dataframes

train = pd.read_csv('../input/train.csv')



print(train.shape)



# X armazena dos dados em um dataframe

X = train.iloc[:,2:]

# y armazena os labels em um dataframe

y = train.iloc[:,1:2]



# target_names armazena os valores distintos dos labels

target_names = train['target'].unique()



# Normaliza os dados de treinamento

scaler = MinMaxScaler()

scaler.fit(X)

X_scaled = scaler.transform(X)



print(X_scaled)
X.head()
X.columns
print("Número de Colunas: ", X.shape[1])
y.head()
# Criação do AutoEncoder com 3 neurônios na camada escondida usando Keras.

#input_dim = X_scaled.shape[1]



# Definição do número de variáveis resultantes do Encoder

#encoding_dim = 10



#input_data = Input(shape=(input_dim,))



# Configurações do Encoder

#encoded = Dense(encoding_dim, activation='linear')(input_data)

#encoded = Dense(encoding_dim, activation='sgmoid')(input_data)

#encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_data)



#encoded1 = Dense(20, activation = 'relu')(input_data)

#encoded2 = Dense(10, activation = 'relu')(encoded1)

#encoded3 = Dense(5, activation = 'relu')(encoded2)

#encoded4 = Dense(encoding_dim, activation = 'relu')(encoded3)



# Configurações do Decoder

#decoded = Dense(input_dim, activation='linear')(encoded)

#decoded = Dense(input_dim, activation='sgmoid')(encoded)



#decoded1 = Dense(5, activation = 'relu')(encoded4)

#decoded2 = Dense(10, activation = 'relu')(decoded1)

#decoded3 = Dense(20, activation = 'relu')(decoded2)

#decoded4 = Dense(input_dim, activation = 'sigmoid')(decoded3)



# Combinando o Encoder e o Decoder em um modelo AutoEncoder

#autoencoder = Model(input_data, decoded4)

#autoencoder.compile(optimizer='adam', loss='mse')

#print(autoencoder.summary())

# Treinamento de fato - Definição de alguns parâmetros como número de épocas, batch size, por exemplo.

#history = autoencoder.fit(X_scaled, X_scaled, epochs=30, batch_size=256, shuffle=True, validation_split=0.1, verbose = 1)



#plot our loss 

#plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

#plt.title('Model Train vs Validation Loss')

#plt.ylabel('Loss')

#plt.xlabel('Epoch')

#plt.legend(['Train', 'Validation'], loc='upper right')

#plt.show()
#test = pd.read_csv('../input/test.csv')



#print(test.shape)



# X armazena dos dados em um dataframe

#X = test.iloc[:,1:]



# Normaliza os dados de treinamento

#scaler = MinMaxScaler()

#scaler.fit(X)

#X_scaled = scaler.transform(X)



# Utilizar o Encoder para codificar os dados de entrada

#encoder = Model(input_data, encoded4)

#encoded_data = encoder.predict(X_scaled)



#print(encoded_data)
# Carregamento das bases de treinamento e teste em dataframes

#train = pd.read_csv('../input/train.csv')

#X = train.drop(["id","target"], axis = 1)

#y = train["target"]



# Separando um conjunto para avaliar

#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)



# Normalização Min-Max

#scaler = MinMaxScaler()

#scaler.fit(X_train)

#X_train = scaler.transform(X_train)

#scaler = MinMaxScaler()

#scaler.fit(X_test)

#X_test = scaler.transform(X_test)



# aplicando autoencoder nos dados de treinamento e teste

#encoder = Model(input_data, encoded4)

#encoded_data_train = encoder.predict(X_train)

#encoder = Model(input_data, encoded4)

#encoded_data_test = encoder.predict(X_test)



# Modelos

# Regressão Logística

#lr = LogisticRegression(penalty='l2', random_state=1)

#lr.fit(encoded_data_train, y_train)

#prob_lr = lr.predict_proba(encoded_data_test)[:,1]

#print("--------------------------------------------------------------------------------------------")

#print("Índice de Gini normalizado para a Regressão Logística com autoencoder: ",gini_normalized(y_test, prob_lr))

#print("--------------------------------------------------------------------------------------------")





# Random Forest com 200 árvores

#rf = RandomForestClassifier(n_estimators = 200, max_depth = 5, random_state = 1, max_features = 7)

#rf.fit(encoded_data_train, y_train)

#prob_rf = rf.predict_proba(encoded_data_test)[:,1]

#print("--------------------------------------------------------------------------------------------")

#print("Índice de Gini normalizado para o Random Forest com autoencoder: ", gini_normalized(y_test, prob_rf))

#print("--------------------------------------------------------------------------------------------")





# XGBoost 

#import xgboost as xgb

#xgbm = xgb.XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.05, random_state = 1)

#xgbm.fit(encoded_data_train, y_train)

#prob_xgb = xgbm.predict_proba(encoded_data_test)[:,1]

#print("--------------------------------------------------------------------------------------------")

#print("Índice de Gini normalizado para o XGBoost com autoencoder: ", gini_normalized(y_test, prob_xgb))

#print("--------------------------------------------------------------------------------------------")





# LightGBM

#from lightgbm import LGBMClassifier

#lgb = LGBMClassifier(n_estimators = 100, learning_rate = 0.02, subsample = 0.7, num_leaves = 15, seed = 1)

#lgb.fit(encoded_data_train, y_train)

#prob_lgb = lgb.predict_proba(encoded_data_test)[:, 1]

#print("--------------------------------------------------------------------------------------------")

#print("Índice de Gini normalizado para o LightGBM com autoencoder: ", gini_normalized(y_test, prob_lgb))

#print("--------------------------------------------------------------------------------------------")
#import warnings

#warnings.filterwarnings('ignore')

#import numpy as np

#import pandas as pd

#from datetime import datetime

#from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#from sklearn.metrics import roc_auc_score

#from sklearn.model_selection import StratifiedKFold

#from sklearn.model_selection import train_test_split

#from xgboost import XGBClassifier
#def timer(start_time=None):

#    if not start_time:

#        start_time = datetime.now()

#        return start_time

#    elif start_time:

#        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

#        tmin, tsec = divmod(temp_sec, 60)

#        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))





# Carregamento das bases de treinamento e teste em dataframes

#train = pd.read_csv('../input/train.csv', dtype={'id': np.int32, 'target': np.int8})

#X = train.drop(["id","target"], axis = 1)

#Y = train["target"].values



# Separando um conjunto para avaliar

#X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state=0)



######################################



#train_df = pd.read_csv('../input/train.csv', dtype={'id': np.int32, 'target': np.int8})

#Y = train_df['target'].values

#X = train_df.drop(['target', 'id'], axis=1)

#test_df = pd.read_csv('../input/test.csv', dtype={'id': np.int32})

#test = test_df.drop(['id'], axis=1)
# Grid de Parâmetros para o XGBoost

#params = {

    #    'min_child_weight': [1, 5, 10],

   #     'gamma': [0.5, 1, 1.5, 2, 5],

  #      'subsample': [0.6, 0.8, 1.0],

 #       'colsample_bytree': [0.6, 0.8, 1.0],

#        'max_depth': [3, 4, 5]}
#xgb = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='binary:logistic', silent=True, nthread=1)
# Número de folds para Cross-Validation

#folds = 2



# Número de combinações a serem feitos no Grid Search. No Total podem ser feitas 3x5x3x3x3 = 405 combinações. Quantos mais combinações, mais tempo leva.

#param_comb = 1



# Configuração dos folds estratificados para o Cross-Validation

#skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)



# Configuração do Grid Search

#random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001 )

#grid = GridSearchCV(estimator=xgb, param_grid=params ,scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3)



# Execução do Treinamento com Grid Search

#start_time = timer(None) # timing starts from this point for "start_time" variable

#random_search.fit(X, Y)

#grid.fit(X, Y)

#timer(start_time) # timing ends here for "start_time" variable
#print('\n All results:')

#print(random_search.cv_results_)

#print('\n Best estimator:')

#print(random_search.best_estimator_)

#print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))

#print(random_search.best_score_ * 2 - 1)

#print('\n Best hyperparameters:')

#print(random_search.best_params_)

#results = pd.DataFrame(random_search.cv_results_)

#results.to_csv('xgb-random-grid-search-results-01.csv', index=False)



# print('\n All results:')

# print(grid.cv_results_)

# print('\n Best estimator:')

# print(grid.best_estimator_)

# print('\n Best score:')

# print(grid.best_score_ * 2 - 1)

# print('\n Best parameters:')

# print(grid.best_params_)

# results = pd.DataFrame(grid.cv_results_)

# results.to_csv('xgb-grid-search-results-01.csv', index=False)



# y_test = grid.best_estimator_.predict_proba(test)

# results_df = pd.DataFrame(data={'id':test_df['id'], 'target':y_test[:,1]})

# results_df.to_csv('submission-grid-search-xgb-porto-01.csv', index=False)
# predict_proba já utiliza o modelo com os melhores hyperparâmetros para realizar o predict da base de teste.

#y_test = random_search.predict_proba(test)



# Em results_df está a base de teste escorada, a coluna target possui as probabilidades

#results_df = pd.DataFrame(data={'id':test_df['id'], 'target':y_test[:,1]})

#print(results_df)

#results_df.to_csv('submission-random-grid-search-xgb-porto-01.csv', index=False)
# Carregando Bibliotecas

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

pd.set_option('display.max_columns', 100)



# Computing gini coefficient ( Coursey Kaggle)

# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation @jit

#def eval_gini(y_true, y_prob):

#    y_true = np.asarray(y_true)

#    y_true = y_true[np.argsort(y_prob)]

#    ntrue = 0

#    gini = 0

#    delta = 0

#    n = len(y_true)

#    for i in range(n-1, -1, -1):

#        y_i = y_true[i]

#        ntrue += y_i

#        gini += y_i * delta

#        delta += 1 - y_i

#    gini = 1 - 2 * gini / (ntrue * (n - ntrue))

#    return gini
# IMPORTAÇÃO DOS DADOS

#train = pd.read_csv('../input/train.csv')

#test = pd.read_csv('../input/test.csv')
# Separando X e y

#X = train.drop(["id","target"], axis = 1)

#y = train["target"].values
# Excluindo variáveis com 'calc'

#colunas = X.columns.tolist()

#colunas_cat = [col for col in colunas if 'cat' in col]

#colunas_calc = [col for col in colunas if 'calc' in col]



#variaveis_excluir = colunas_calc

#X.drop(variaveis_excluir, inplace=True, axis=1)
# One-hot nas categóricas

#X = pd.get_dummies(X)

#X.head()
# Separando um conjunto para avaliar

#X_train, X_test, y_train, y_test = train_test_split(

#    X, y, stratify=y, random_state=0)

# Normalização Min-Max

#scaler = MinMaxScaler()

#scaler.fit(X_train)

#X_train = scaler.transform(X_train)

#scaler = MinMaxScaler()

#scaler.fit(X_test)

#X_test = scaler.transform(X_test)
# Testar os modelos

# Modelos

# Regressão Logística

# Testando com a regressão logística com penalização L2

#lr = LogisticRegression(penalty='l2', random_state=1)

#lr.fit(X_train, y_train)

#prob_lr = lr.predict_proba(X_test)[:,1]

#print("--------------------------------------------------------------------------------------------")

#print("Índice de Gini normalizado para a Regressão Logística: ",eval_gini(y_test, prob_lr))

#print("--------------------------------------------------------------------------------------------")

# Random Forest com 200 árvores

#rf = RandomForestClassifier(n_estimators = 200, max_depth = 4, random_state = 1, max_features = 15)

#rf.fit(X_train, y_train)

#prob_rf = rf.predict_proba(X_test)[:,1]

#print("--------------------------------------------------------------------------------------------")

#print("Índice de Gini normalizado para o Random Forest: ", eval_gini(y_test, prob_rf))

#print("--------------------------------------------------------------------------------------------")

# XGBoost 

#xgbm = XGBClassifier(max_depth=4, n_estimators=100, learning_rate=0.05, random_state = 1)

#xgbm.fit(X_train, y_train)

#prob_xgb = xgbm.predict_proba(X_test)[:,1]

#print("--------------------------------------------------------------------------------------------")

#print("Índice de Gini normalizado para o XGBoost: ", eval_gini(y_test, prob_xgb))

#print("--------------------------------------------------------------------------------------------")

# LightGBM

#lgb = LGBMClassifier(n_estimators = 100, learning_rate = 0.02, subsample = 0.7, num_leaves = 15, seed = 1)

#lgb.fit(X_train, y_train)

#prob_lgb = lgb.predict_proba(X_test)[:, 1]

#print("--------------------------------------------------------------------------------------------")

#print("Índice de Gini normalizado para o LightGBM: ", eval_gini(y_test, prob_lgb))

#print("--------------------------------------------------------------------------------------------")
## Testando na base de treinamento completa

# Normalização Min-Max

#scaler = MinMaxScaler()

#scaler.fit(X)

#X = scaler.transform(X)
# base completa

# Modelos

#prob_lr = lr.predict_proba(X)[:,1]

#print("--------------------------------------------------------------------------------------------")

#print("Índice de Gini normalizado para a Regressão Logística: ",eval_gini(y, prob_lr))

#print("--------------------------------------------------------------------------------------------")

#prob_rf = rf.predict_proba(X)[:,1]

#print("--------------------------------------------------------------------------------------------")

#print("Índice de Gini normalizado para o Random Forest: ", eval_gini(y, prob_rf))

#print("--------------------------------------------------------------------------------------------")

#prob_xgb = xgbm.predict_proba(X)[:,1]

#print("--------------------------------------------------------------------------------------------")

#print("Índice de Gini normalizado para o XGBoost: ", eval_gini(y, prob_xgb))

#print("--------------------------------------------------------------------------------------------")

#prob_lgb = lgb.predict_proba(X)[:, 1]

#print("--------------------------------------------------------------------------------------------")

#print("Índice de Gini normalizado para o LightGBM: ", eval_gini(y, prob_lgb))

#print("--------------------------------------------------------------------------------------------")
# Criando o arquivo para submissão

#test_df = pd.read_csv('../input/test.csv', dtype={'id': np.int32})

#test = test_df.drop(['id'], axis=1)

#test.drop(variaveis_excluir, inplace=True, axis=1)

#test = pd.get_dummies(test)

#scaler = MinMaxScaler()

#scaler.fit(test)

#test = scaler.transform(test)

#y_test = xgbm.predict_proba(test)[:,1]



# Em results_df está a base de teste escorada, a coluna target possui as probabilidades

#results_df = pd.DataFrame(data={'id':test_df['id'], 'target':y_test})

#print(results_df)

#results_df.to_csv('submission-random-grid-search-xgb-01.csv', index=False)