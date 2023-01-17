# Imports

# Manipulação de dados

import numpy as np

import pandas as pd

from scipy import stats

# Gráficos

import matplotlib.pyplot as plt

import seaborn as sns

#Preparação

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

# Cross Validation

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

# Modelo

from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

# warning

import warnings

warnings.filterwarnings("ignore")
# Carregando os dados

X_treino = pd.read_csv('../input/competicao-dsa-machine-learning-sep-2019/X_treino.csv')

X_teste = pd.read_csv('../input/competicao-dsa-machine-learning-sep-2019/X_teste.csv')

y_treino = pd.read_csv('../input/competicao-dsa-machine-learning-sep-2019/y_treino.csv')
# Adicionado a variável target ao banco de treino

data_train = pd.merge(X_treino, y_treino, on = 'series_id')
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

   

    return summary
# Resumo dos dados

resumetable(data_train)
# Distribuição das classes

y_treino.groupby('surface').size()
# Gráfico da variável target

plt.figure(figsize = (16,6))

freq = len(y_treino)



g = sns.countplot(x = 'surface', data = y_treino, order = y_treino['surface'].value_counts().index)

g.set_xlabel("Surface", fontsize = 18)

g.set_ylabel("Quantidade", fontsize = 18)



for p in g.patches:

    height = p.get_height()

    g.text(p.get_x() + p.get_width()/2., height + 3,

          '{:1.2f}%'.format(height/freq * 100),

          ha = "center", fontsize = 18)
# Criando uma nova variável 'surface_cod'

le = preprocessing.LabelEncoder()

data_train['surface_cod'] = le.fit_transform(data_train['surface'])
data_train.describe()
# Plot para distirbuições bi-variadas

dados = data_train[['orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W', 'angular_velocity_X', 'angular_velocity_Y',

                  'angular_velocity_Z', 'linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z', 'surface_cod']]

colunas = ['orien_X', 'orien_Y', 'orien_Z', 'orient_W', 'ang_velo_X', 'ang_velo_Y',

                  'ang_velo_Z', 'lin_acc_X', 'lin_acc_Y', 'lin_acc_Z', 'sur_cod']



# Matriz de Correlação com nomes das variáveis

correlations = dados.corr()



# Plot

#import numpy as np

fig = plt.figure(figsize = (20,12))

plt.suptitle('Matriz de Correlação', fontsize=22)

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin = -1, vmax = 1)

fig.colorbar(cax)

ticks = np.arange(0, 11, 1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(colunas)

ax.set_yticklabels(colunas)

plt.show()
# Distribuição

plt.figure(figsize = (20,7))

plt.suptitle('Distribuição de valores das variáveis preditoras', fontsize=22)

plt.subplot(241)

g1 = sns.distplot(data_train['orientation_X'])

g1.set_xlabel("orientation_X", fontsize = 15)



plt.subplot(242)

g2 = sns.distplot(data_train['orientation_Y'])

g2.set_xlabel("orientation_Y", fontsize = 15)



plt.subplot(243)

g3 = sns.distplot(data_train['orientation_Z'])

g3.set_xlabel("orientation_Z", fontsize = 15)



plt.subplot(244)

g4 = sns.distplot(data_train['orientation_W'])

g4.set_xlabel("orientation_W", fontsize = 15)



plt.figure(figsize = (20,7))



plt.subplot(231)

g5 = sns.distplot(data_train['angular_velocity_X'])

g5.set_xlabel("angular_velocity_X", fontsize = 15)



plt.subplot(232)

g6 = sns.distplot(data_train['angular_velocity_Y'])

g6.set_xlabel("angular_velocity_Y", fontsize = 15)



plt.subplot(233)

g7 = sns.distplot(data_train['angular_velocity_Z'])

g7.set_xlabel("angular_velocity_Z", fontsize = 15)



plt.figure(figsize = (20, 7))



plt.subplot(231)

g8 = sns.distplot(data_train['linear_acceleration_X'])

g8.set_xlabel("linear_acceleration_X", fontsize = 15)



plt.subplot(232)

g9 = sns.distplot(data_train['linear_acceleration_Y'])

g9.set_xlabel("linear_acceleration_Y", fontsize = 15)



plt.subplot(233)

g10 = sns.distplot(data_train['linear_acceleration_Z'])

g10.set_xlabel("linear_acceleration_Z", fontsize = 15)
# Dados de treino

X_train = data_train.drop(['row_id', 'series_id', 'measurement_number', 'group_id', 'surface', 'surface_cod'],axis=1)



# Normalizando os dados

X_train = MinMaxScaler().fit_transform(X_train)



# Padronizando os dados

X_train = StandardScaler().fit_transform(X_train)
# Transformando os dados

# Carregando os dados



Y_train = data_train['surface_cod']



# Definindo os valores para o número de folds

num_folds = 10

seed = 7



# Separando os dados em folds

kfold = KFold(num_folds, True, random_state = seed)



# Criando o modelo classificador

cart = DecisionTreeClassifier()



# Definindo o número de trees

num_trees = 10



# Criando o modelo bagging

modelo = BaggingClassifier(base_estimator = cart, n_estimators = num_trees, random_state = seed).fit(X_train, Y_train)



# Cross Validation

resultado = cross_val_score(modelo, X_train, Y_train, cv = kfold)
# Dados de teste

X_test = X_teste.drop(['row_id', 'series_id', 'measurement_number'],axis=1)



# Normalizando os dados

X_test = MinMaxScaler().fit_transform(X_test)



# Padronizando os dados

X_test = StandardScaler().fit_transform(X_test)



# Previsões com os dados de teste

pred = modelo.predict(X_test)
# Criando o data frame para a subimissão

df_pred = pd.DataFrame({"series_id": X_teste.series_id, "surface": pred})

df_pred['surface'] = le.inverse_transform(df_pred['surface'])

df_pred_final = df_pred.drop_duplicates('series_id')
# salvando a previsão

df_pred_final.to_csv('arq_submission.csv', index=False)