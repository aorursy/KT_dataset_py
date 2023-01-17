# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.



import os

print(os.listdir("../input"))
# Importar base de treino

df_train = pd.read_csv("../input/train.csv") 



# Importar a base de teste

df_test = pd.read_csv("../input/test.csv")
# Verificar a dimensão das bases

df_train.shape, df_test.shape
# Unir as bases

df = pd.concat([df_train, df_test])
df.shape
# Verificar as bases importadas

df.head().T
# Verificar se bases são Data Frame

type(df_train),type(df_test), type(df)
# Medidas resumo das variáveis

df.describe().T
# Verificar tipo de colunas das bases  (OU df_train.dtypes)

df_train.info(),df_test.info(), df.info()
#  Somar registros faltantes (NaN ou nulo) no campo nota_mat.

df['nota_mat'].isnull().sum()
# Avaliar número de valores diferentes por coluna

df.nunique()
df.info()
# Tratar os caracteres especiais ou que gerem erro nas variáveis

# e converter variável string para numérica

df['area'] = df['area'].str.replace(',','').astype(float)
df['codigo_mun'].head()
df['codigo_mun'] = df['codigo_mun'].str.replace('ID_ID_','').astype(int)
df['densidade_dem'] = df['densidade_dem'].str.replace(',','').astype(float)
df['ranking_igm'] = df_train['ranking_igm'].str.replace('º','').astype(float)
df['servidores'] = df['servidores'].fillna(df['servidores'].min())
df['comissionados_por_servidor'] = (df['comissionados']/df['servidores']*100).astype(float, inplace= True)
#  Somar registros faltantes (NaN ou nulo) na base.

df.isnull().sum()
# TRATAR OS VALORES NaN

df['densidade_dem'] = df['densidade_dem'].fillna(0)
df['exp_anos_estudo'] =df['exp_anos_estudo'].fillna(df['exp_anos_estudo'].mean())

df['exp_vida'] =df['exp_vida'].fillna(df['exp_vida'].mean())

df['exp_vida'] =df['exp_vida'].fillna(df['exp_vida'].mean())

df['idhm'] =df['idhm'].fillna(df['idhm'].mean())
df['gasto_pc_educacao'] = df['gasto_pc_educacao'].fillna(0)

df['gasto_pc_saude'] = df['gasto_pc_saude'].fillna(0)

df['hab_p_medico'] = df['hab_p_medico'].fillna(0)

df['indice_governanca'] = df['indice_governanca'].fillna(0)

df['participacao_transf_receita'] = df['participacao_transf_receita'].fillna(0)

df['perc_pop_econ_ativa'] = df['perc_pop_econ_ativa'].fillna(0)
df['ranking_igm'] = df['ranking_igm'].fillna(0)
#  Somar registros faltantes (NaN ou nulo) na base.

df.isnull().sum()
df.head()
# ANÁLISE EXPLORATÓRIA
df.head().T
# Medidas resumo das variáveis

df.describe().T
# Verificar tipo de colunas das bases  (OU df_train.dtypes)

df_train.info(),df_test.info(), df.info()
# Amostra dos cinco primeiros registros da base de treino

df_train.sample(5)
# Amostra dos cinco primeiros registros da base de treino

df_test.sample(5)
#df['exp_anos_estudo'].value_counts().plot.bar()

#df['exp_anos_estudo'].plot.bar()
# Para avaliar a distribuição dos valores - Histograma e Boxplot

df.hist()
df['nota_mat'].plot.box()
pd.crosstab(df['regiao'],df['exp_vida'])
import seaborn as sns
sns.distplot(np.exp(df_train['nota_mat']), bins=10)
sns.boxplot(y='nota_mat', data=df_train)
sns.boxplot(y='nota_mat', x='regiao', data=df_train)
sns.violinplot(y='nota_mat', data=df_train)
sns.violinplot(y='nota_mat', x='regiao', data=df_train)
# Fazer análises agrupadas

df_train.groupby('estado')['nota_mat'].mean()
# Boxplot da expectativa de anos de estudo por estado



%matplotlib inline

plt.figure(figsize=(25,20))

sns.boxplot(x='estado', y='exp_anos_estudo', data=df)
%matplotlib inline

plt.figure(figsize=(25,20))

sns.boxplot(x='regiao', y='nota_mat', data=df)
df.groupby('regiao')['nota_mat'].mean()
#df.groupby('estado')['nota_mat'].mean().plot.bar()



%matplotlib inline

plt.figure(figsize=(25,20))

sns.boxplot(x='estado', y='nota_mat', data=df)
sns.barplot(y='nota_mat', x='regiao', hue='capital', data=df)
df.groupby('estado')['nota_mat'].mean().plot.barh()
# Análise Exploratória de Grupos

# .describe() após um .groupby()



df.groupby('estado')['nota_mat'].describe()
df_train.groupby('estado')['nota_mat'].describe()['mean'].sort_index(ascending=False).plot()
df_train.groupby('estado')['nota_mat'].describe()['50%'].sort_index(ascending=False).plot()
# Variáveis contínuas

sns.pairplot(x_vars='exp_vida', y_vars='exp_anos_estudo', data=df, size=7)
sns.pairplot(x_vars='exp_vida', y_vars='exp_anos_estudo', hue='regiao', data=df, size=7)
sns.pairplot(x_vars='exp_vida', y_vars='exp_anos_estudo', hue='regiao', data=df, size=7, kind='reg')
# Correlação para dataframe

df.corr()
plt.figure(figsize=(30,30))

sns.heatmap(df.corr(), annot=True, cmap="Blues")
# valores em branco no DataFrame

df.info()
# Verificar frequencia das variávies categóricas
df['regiao'].value_counts()
df['porte'].value_counts()
df['estado'].value_counts()
# Dummificar Variáveis regiao e porte

#df = pd.concat([df, pd.get_dummies(df['regiao'], prefix='reg').iloc[:, :-1]], 

#                   axis =1)
#Dada uma matriz X, filtramos as variáveis

#categóricas em uma nova matriz categ_X

 

#encoder = preprocessing.OneHotEncoder()

 

#encoder.fit(categ_X)

#categ_X = encoder.transform(categ_X)

 

#Realizando a junção

#X = np.append(X, categ_X, axis=1)
#Realizando a junção

#X = np.append(X, categ_X, axis=1)
# Dummificar Variáveis regiao e porte

dm_regiao = pd.get_dummies(df['regiao'], prefix='reg')
dm_regiao.head()
dm_regiao.info()
dm_regiao[['reg_CENTRO-OESTE','reg_NORDESTE','reg_NORTE','reg_SUDESTE','reg_SUL']]=dm_regiao[['reg_CENTRO-OESTE','reg_NORDESTE','reg_NORTE','reg_SUDESTE','reg_SUL']].astype(int)
dm_regiao.info()
#Realizando a junção

#df = df.append(dm_regiao,axis=0)

df = pd.concat([df, dm_regiao], axis = 1)
dm_porte = pd.get_dummies(df['porte'], prefix='p')
dm_porte.head()
dm_porte[['p_Grande porte','p_Médio porte','p_Pequeno porte 1','p_Pequeno porte 2']]=dm_porte[['p_Grande porte','p_Médio porte','p_Pequeno porte 1','p_Pequeno porte 2']].astype(int)
dm_porte.info()
df = pd.concat([df, dm_porte], axis = 1)
df.shape
df.info()
#  Somar registros faltantes (NaN ou nulo) na base.

df.isnull().sum()
# Separar as bases de treino e teste após tratamento

treino = df[~df['nota_mat'].isnull()]

test   = df[df['nota_mat'].isnull()]
treino.shape, test.shape
treino['nota_mat'] = np.log(treino['nota_mat'])
treino['nota_mat'].head()
# Separando (base df_train -> treino) em treino e validação

from sklearn.model_selection import train_test_split
df_treino, df_valid = train_test_split(treino, random_state=42)
df_treino.shape, df_valid.shape
# test = df_train[df_train['nota_mat'].isnull()]
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR
models = {'RandomForest': RandomForestRegressor(random_state=42),

         'ExtraTrees': ExtraTreesRegressor(random_state=42),

         'GBM': GradientBoostingRegressor(random_state=42),

         'DecisionTree': DecisionTreeRegressor(random_state=42),

         'AdaBoost': AdaBoostRegressor(random_state=42),

         'KNN 1': KNeighborsRegressor(n_neighbors=1),

         'KNN 3': KNeighborsRegressor(n_neighbors=3),

         'KNN 11': KNeighborsRegressor(n_neighbors=11),

         'SVR': SVR(),

         'Linear Regression': LinearRegression()}
# removed_cols = ['regiao', 'estado', 'municipio', 'nota_mat', 'porte','Unnamed: 0', 'codigo_mun']

removed_cols = ['regiao', 'estado', 'municipio', 'nota_mat', 'porte','Unnamed: 0', 'codigo_mun']
feats = [c for c in df_treino.columns if c not in removed_cols]
from sklearn.metrics import mean_squared_error
def run_model(model, df_treino, df_valid, feats, y_name):

    model.fit(df_treino[feats], df_treino[y_name])

    preds = model.predict(df_valid[feats])

    return mean_squared_error(df_valid[y_name], preds)**(1/2)

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
scores = []

for name, model in models.items():

    score = run_model(model, df_treino, df_valid, feats, 'nota_mat')

    scores.append(score)

    print(name+':', score)
pd.Series(scores, index=models.keys()).sort_values(ascending=False).plot.barh()
df_treino['preds'] = df_treino['nota_mat'].mean()
df_treino.shape, df_treino['nota_mat'].mean()
mean_squared_error(df_treino['nota_mat'], df_treino['preds'])
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(df_treino[feats], df_treino['nota_mat'])
train_preds = rf.predict(df_treino[feats])
train_preds
mean_squared_error(df_treino['nota_mat'], train_preds)**(1/2)
valid_preds = rf.predict(df_valid[feats])
mean_squared_error(df_valid['nota_mat'], valid_preds)**(1/2)
#test['nota_mat'] = np.exp(rf.predict(test[feats]))

test['nota_mat'] = np.exp(rf.predict(test[feats]))
pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
# np.exp(test.groupby('estado')['nota_mat'].mean()).plot.bar()
test.head()
test.head()
np.exp(df_treino['nota_mat']).mean(), np.exp(df_valid['nota_mat']).mean(), test['nota_mat'].mean()
test.groupby('regiao')['nota_mat'].mean().plot.bar()
test[['codigo_mun','nota_mat']].to_csv('rf.csv', index=False)
mean_squared_error(np.exp(df_treino['nota_mat']), train_preds)**(1/2)
#%matplotlib notebook

df.boxplot(column='nota_mat', by='regiao')
#%matplotlib notebook

#matplotlib.style.use('ggplot')

df.boxplot(column='nota_mat')