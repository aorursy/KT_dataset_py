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

df = pd.read_csv('../input/train.csv')



df_energia = pd.read_csv('../input/potential_energy.csv')

df_charges = pd.read_csv('../input/mulliken_charges.csv')

df_estrutura = pd.read_csv('../input/structures.csv')



df.head(5)
df_teste = pd.read_csv('../input/test.csv')
# Mostrando as estruturas dos Datasets

df.info()
# Mostrando as estruturas dos Datasets - Energia

df_energia.info()
# Mostrando as estruturas dos Datasets - Estrutura

df_estrutura.info()
#Checando valores NA nos dados

df.isna().any()[lambda x: x]
df_teste.isna().any()[lambda x: x]
df_estrutura.isna().any()[lambda x: x]
df_energia.isna().any()[lambda x: x]
# Coletando os dados - Merge / Mesclagem dos DataSets

df = pd.merge(df, df_energia, how = 'left',

                  left_on  = ['molecule_name'],

                  right_on = ['molecule_name'])



df = pd.merge(df, df_charges, how = 'left',

                  left_on  = ['molecule_name','atom_index_0'],

                  right_on = ['molecule_name','atom_index'])



df = df.drop('atom_index', axis=1)

df = pd.merge(df, df_charges, how = 'left',

                  left_on  = ['molecule_name','atom_index_1'],

                  right_on = ['molecule_name','atom_index'])

df = df.drop('atom_index', axis=1)

df.head(10)
#Replicando para os dados de Teste

df_teste = pd.merge(df_teste, df_energia, how = 'left',

                  left_on  = ['molecule_name'],

                  right_on = ['molecule_name'])



df_teste = pd.merge(df_teste, df_charges, how = 'left',

                  left_on  = ['molecule_name','atom_index_0'],

                  right_on = ['molecule_name','atom_index'])



df_teste = df_teste.drop('atom_index', axis=1)



df_teste = pd.merge(df_teste, df_charges, how = 'left',

                  left_on  = ['molecule_name','atom_index_1'],

                  right_on = ['molecule_name','atom_index'])



df_teste = df_teste.drop('atom_index', axis=1)
# Coletando os dados - Merge / Mesclagem dos DataSets - Parte 2

df = pd.merge(df, df_estrutura, how = 'left',

                  left_on  = ['molecule_name','atom_index_0'],

                  right_on = ['molecule_name','atom_index'])



df = df.rename(columns={'atom': 'atom_0',

                            'x': 'x_0',

                            'y': 'y_0',

                            'z': 'z_0'})



df = df.drop('atom_index', axis=1)



df = pd.merge(df, df_estrutura, how = 'left',

                 left_on  = ['molecule_name','atom_index_1'],

                 right_on = ['molecule_name','atom_index'])



df = df.rename(columns={'atom': 'atom_1',

                            'x': 'x_1',

                            'y': 'y_1',

                            'z': 'z_1'})



df = df.drop('atom_index', axis=1)



df.head(10)
# Replicando - Parte 2 para o Teste

df_teste = pd.merge(df_teste, df_estrutura, how = 'left',

                  left_on  = ['molecule_name','atom_index_0'],

                  right_on = ['molecule_name','atom_index'])



df_teste = df_teste.rename(columns={'atom': 'atom_0',

                            'x': 'x_0',

                            'y': 'y_0',

                            'z': 'z_0'})



df_teste = df_teste.drop('atom_index', axis=1)



df_teste = pd.merge(df_teste, df_estrutura, how = 'left',

                 left_on  = ['molecule_name','atom_index_1'],

                 right_on = ['molecule_name','atom_index'])



df_teste = df_teste.rename(columns={'atom': 'atom_1',

                            'x': 'x_1',

                            'y': 'y_1',

                            'z': 'z_1'})



df_teste = df_teste.drop('atom_index', axis=1)
df.dtypes
# Criando nova coluna do type

df['type1'] = df['type'].apply(lambda x: x[0])

df_teste['type1'] = df_teste['type'].apply(lambda x: x[0])
df.head(10)
df['atom_0'].unique()
df['atom_1'].unique()
# Radius dos átomos - Nova Coluna

df['rad'] = df['atom_0'].map({'H':0.43, 'C':0.82, 'N':0.8, 'O':0.78, 'F':0.76})

df['rad_1'] = df['atom_1'].map({'H':0.43, 'C':0.82, 'N':0.8, 'O':0.78, 'F':0.76})



# Nova coluna de Eletrons negativos

df['electro'] = df['atom_0'].map({'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98})

df['electro1'] = df['atom_1'].map({'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98})

df_teste.dtypes
# Radius dos átomos - Nova Coluna - TESTE

df_teste['rad'] = df_teste['atom_0'].map({'H':0.43, 'C':0.82, 'N':0.8, 'O':0.78, 'F':0.76})

df_teste['rad_1'] = df_teste['atom_1'].map({'H':0.43, 'C':0.82, 'N':0.8, 'O':0.78, 'F':0.76})



# Nova coluna de Eletrons negativos  - TESTE

df_teste['electro'] = df_teste['atom_0'].map({'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98})

df_teste['electro1'] = df_teste['atom_1'].map({'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98})
df.head(5)
#Limpando a Memória

del([df_energia, df_charges, df_estrutura])

# Calculando a distancia entre os 2 atómos - Nova coluna

df["distance"] = ((df['x_0']-df['x_1'])**2 + (df['y_0']-df['y_1'])**2 + (df['z_0']-df['z_1'])**2) ** (1/2)

# Calculando o mulliken entre os 2 atómos - Nova coluna

df['charge'] = ((df['mulliken_charge_x'] - df['mulliken_charge_y']) **2) ** (1/2)
# Calculando a distancia entre os 2 atómos - Nova coluna - TESTE

df_teste["distance"] = ((df_teste['x_0']-df_teste['x_1'])**2 + (df_teste['y_0']-df_teste['y_1'])**2 + (df_teste['z_0']-df_teste['z_1'])**2) ** (1/2)

# Calculando o mulliken entre os 2 atómos - Nova coluna

df_teste['charge'] = ((df_teste['mulliken_charge_x'] - df_teste['mulliken_charge_y']) **2) ** (1/2)
#Calculando a distancia por molécula individualmente e calculo a diferença entre as duas

df["distance_0"] = 1 / (((df['x_0']**2) + (df['y_0']**2) + (df['z_0']**2)) ** (1/2) ** 3 )

df["distance_1"] = 1 / (((df['x_1']**2) + (df['y_1']**2) + (df['z_1']**2)) ** (1/2) ** 3 )

df["distance_dif"] = (df["distance_0"] * df["distance_1"]) / (df["distance_0"] + df["distance_1"])



#Calculando a distancia por molécula individualmente e calculo a diferença entre as duas utilizando o radios das moléculas

df["distance_0_Rad"] = 1/((df["distance_0"]-df["rad"])** (1/2) ** 3 )

df["distance_1_Rad"] = 1/((df["distance_1"]-df["rad_1"])** (1/2) ** 3 )

df["distance_dif_Rad"] = (df["distance_0_Rad"] * df["distance_1_Rad"]) / (df["distance_0_Rad"] + df["distance_1_Rad"]) 



#Calculo dos eletrons

df["distance_0_Elec"] = (df["distance_0"]*df["electro"])** 3

df["distance_1_Elec"] = (df["distance_1"]*df["electro1"])** 3

df["distance_dif_Elec"] = (df["distance_0_Elec"] * df["distance_1_Elec"]) / (df["distance_0_Elec"] + df["distance_1_Elec"]) 

#Replicando para o TESTE

df_teste["distance_0"] = 1 / (((df_teste['x_0']**2) + (df_teste['y_0']**2) + (df_teste['z_0']**2)) ** (1/2) ** 3 )

df_teste["distance_1"] = 1 / (((df_teste['x_1']**2) + (df_teste['y_1']**2) + (df_teste['z_1']**2)) ** (1/2) ** 3 )

df_teste["distance_dif"] = (df_teste["distance_0"] * df_teste["distance_1"]) / (df_teste["distance_0"] + df_teste["distance_1"])



#Calculando a distancia por molécula individualmente e calculo a diferença entre as duas utilizando o radios das moléculas

df_teste["distance_0_Rad"] = 1/((df_teste["distance_0"]-df_teste["rad"])** (1/2) ** 3 )

df_teste["distance_1_Rad"] = 1/((df_teste["distance_1"]-df_teste["rad_1"])** (1/2) ** 3 )

df_teste["distance_dif_Rad"] = (df_teste["distance_0_Rad"] * df_teste["distance_1_Rad"]) / (df_teste["distance_0_Rad"] + df_teste["distance_1_Rad"]) 



#Calculo dos eletrons

df_teste["distance_0_Elec"] = (df_teste["distance_0"]*df_teste["electro"])** 3

df_teste["distance_1_Elec"] = (df_teste["distance_1"]*df_teste["electro1"])** 3

df_teste["distance_dif_Elec"] = (df_teste["distance_0_Elec"] * df_teste["distance_1_Elec"]) / (df_teste["distance_0_Elec"] + df_teste["distance_1_Elec"]) 

#Checando valores NA nos dados

df.isna().any()[lambda x: x]
df_teste.isna().any()[lambda x: x]
#Preenchendo NAs

df['distance_1_Rad'].fillna((df['distance_1_Rad'].mean()), inplace=True)

df['distance_dif_Rad'].fillna((df['distance_dif_Rad'].mean()), inplace=True)

#Preenchendo NAs

df_teste['distance_1_Rad'].fillna((df_teste['distance_1_Rad'].mean()), inplace=True)

df_teste['distance_dif_Rad'].fillna((df_teste['distance_dif_Rad'].mean()), inplace=True)

df_teste['potential_energy'].fillna((df_teste['potential_energy'].mean()), inplace=True)

df_teste['mulliken_charge_x'].fillna((df_teste['mulliken_charge_x'].mean()), inplace=True)

df_teste['mulliken_charge_y'].fillna((df_teste['mulliken_charge_y'].mean()), inplace=True)

df_teste['charge'].fillna((df_teste['charge'].mean()), inplace=True)

# Separando as colunas que não serão usadas no modelo

df = df.drop('id', axis=1)

df = df.drop('molecule_name', axis=1)

df = df.drop('atom_index_0', axis=1)

df = df.drop('atom_index_1', axis=1)

df = df.drop('type', axis=1)

df = df.drop('atom_0', axis=1)

df = df.drop('atom_1', axis=1)

df_teste = df_teste.drop('id', axis=1)

df_teste = df_teste.drop('molecule_name', axis=1)

df_teste = df_teste.drop('atom_index_0', axis=1)

df_teste = df_teste.drop('atom_index_1', axis=1)

df_teste = df_teste.drop('type', axis=1)

df_teste = df_teste.drop('atom_0', axis=1)

df_teste = df_teste.drop('atom_1', axis=1)
df_treino = df
del(df)
# Correlação com a Variavel TARGET => scalar_coupling_constant

df_treino[df_treino.columns.drop('scalar_coupling_constant')].corrwith(df_treino.scalar_coupling_constant)

# Visualizando e analisando os dados



# Construindo um gráfico de HEATMAP

f, ax = plt.subplots(figsize=(15, 12))

sns.heatmap(df_treino.corr(),linewidths=.5, ax=ax)
# Histogramas

#df_treino.plot(kind = 'hist', subplots = True, layout = (7,4), sharex = False, figsize=(20,70))

#plt.show()
# Box-Plot

df_treino.plot(kind = 'box', subplots = True, layout = (7,4), sharex = False, sharey = False, figsize=(20,70))

plt.show()
df_treino.shape
df_teste.shape
# Ajustando e padronizando as escalas - Normalização

from sklearn.preprocessing import MinMaxScaler



array = df_treino.values



# Separando o array em componentes de input (X) e output (Y)

X = array[:,1:26]

X_sub = df_teste.values

Y = array[:,0]

#Limpando a Memória

del(df_teste)
# Gerando a nova escala (normalizando os dados)

scaler = MinMaxScaler(feature_range = (0, 1))

rescaledX = scaler.fit_transform(X)

rescaledX_teste = scaler.fit_transform(X_sub)

# Comecando o modelo - Versao 1 - Mais basico e simples

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression



# Divide os dados em treino e teste

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 5)

# Criando o modelo

modelo = LinearRegression()



# Treinando o modelo

modelo.fit(X_train, Y_train)



# Fazendo previsões

Y_pred = modelo.predict(X_test)



# Resultado

mae = mean_absolute_error(Y_test, Y_pred)

print('Modelo 1 - Regressao Linear => MAE = %0.4f' % mae)
# Modelo 2 - lightgbm simples

import lightgbm as lgb



gbm = lgb.LGBMRegressor(num_leaves=50,

                        learning_rate=0.05,

                        n_estimators=100)

gbm.fit(X_train, Y_train,

        eval_set=[(X_test, Y_test)],

        eval_metric='l1',

        early_stopping_rounds=100,

        verbose=0

       )

Y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)



# Resultado

mae = np.log(mean_absolute_error(Y_test, Y_pred))

print('Modelo 2 - LightGBM => MAE = %0.4f' % mae)
# Funcoes utilitaria

# Treinamento e resultado do modelo - funcao generica

def treine_e_avalie(model, X, y, X_test, y_test):

    

    # Predicao

    model_pred = treino_e_predicao(model, X, y, X_test)

    #Performance

    model_mae = np.log(mean_absolute_error(y_test, model_pred))

    

    # Retorno da Performance do modelo

    return model_mae



def treino_e_predicao(model, X, y, X_test):

    

    # FIT

    model.fit(X, y)

    # Predicao

    return model.predict(X_test)

# Importando mais modelos 

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor
# Modelo 3 - KNN

knn = KNeighborsRegressor(n_neighbors=5)

knn_mae = treine_e_avalie(knn, X_train, Y_train, X_test, Y_test)



print('Modelo 3 - KNN => MAE = %0.4f' % knn_mae)
# Modelo 4 - GradientBoostingRegressor

gradient_boosted = GradientBoostingRegressor(random_state=60)

gradient_boosted_mae = treine_e_avalie(gradient_boosted, X_train, Y_train, X_test, Y_test)



print('Modelo 4 - GradientBoostingRegressor = %0.4f' % gradient_boosted_mae)

#Otimizando o modelo 2

#from sklearn.model_selection import GridSearchCV



#estimator = lgb.LGBMRegressor()



#param_grid = {

#    'learning_rate': [0.005,0.05, 0.1, 1],

#    'n_estimators': [100],

#    'num_leaves': [50],

#    'boosting_type' : ['gbdt','dart'],

#    'objective' : ['regression'],

#    'colsample_bytree' : [0.65, 0.66],

#    'subsample' : [0.7,0.75],

#    'reg_alpha' : [1,1.2],

#    'reg_lambda' : [1,1.2,1.4]    

#}



#gbm1 = GridSearchCV(estimator, param_grid, cv=5, verbose=0)

#gbm1.fit(X_train, Y_train)



#print('Melhores parametros:', gbm1.best_params_)

#print('Melhor score:', gbm1.best_score_)
# Modelo 5 - otimizado

gbm = lgb.LGBMRegressor(num_leaves=100,

                        learning_rate=1.10,

                        n_estimators=500,

                        boosting_type='dart',

                        metric='mae',

                        objective='regression_l1',

                        #max_depth=19,

                        #subsample=0.75,

                        verbosity=-0,

                        reg_alpha=1.2,

                        reg_lambda=1,

                        #sub_feature = 0.75,

                        #sub_row = 0.50,

                        #bagging_freq = 1,                        

                        #colsample_bytree=0.65

                        )

gbm.fit(X_train, Y_train,

        eval_set=[(X_test, Y_test)],

        eval_metric='l1',

        early_stopping_rounds=200,

        verbose=1

       )



Y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)



# Resultado

mae = np.log(mean_absolute_error(Y_test, Y_pred))

print('Modelo 5 - LightGBM - Otimizado => MAE = %0.4f' % mae)
#Features mais importantes do modelo

#sorted(zip(gbm.feature_importances_, df_treino.columns), reverse=True)

#Gerando os dados de submissao e predição

df_submission = pd.read_csv('../input/sample_submission.csv')

resultado = gbm.predict(X_sub, num_iteration=gbm.best_iteration_)

df_submission['scalar_coupling_constant'] = resultado
#Gravando Arquivo de Submissao

df_submission.to_csv('submission.csv', index=False)