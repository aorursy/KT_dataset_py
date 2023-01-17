# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Estude os Dados e Metadados - (2343, 29)

train.shape
# (781, 28)

test.shape
# Não havia a necessidade de utilizar o np.log, porém não faz diferença usar neste caso

train['nota_mat'] = np.log(train['nota_mat'])
df = train.append(test, sort=False)
df.sample(5)
df.count()
for col in df.columns:

    if (df[col].count() < 3124):

        print(col)

        print(3124 - df[col].count())

    
df.nunique()
df.info()
df['participacao_transf_receita'] = df['participacao_transf_receita'].fillna(np.mean(df['participacao_transf_receita']))

df['servidores'] = df['servidores'].fillna(np.mean(df['servidores']))

df['perc_pop_econ_ativa'] = df['perc_pop_econ_ativa'].fillna(np.mean(df['perc_pop_econ_ativa']))

df['gasto_pc_saude'] = df['gasto_pc_saude'].fillna(np.mean(df['gasto_pc_saude']))

df['hab_p_medico'] = df['hab_p_medico'].fillna(np.mean(df['hab_p_medico']))

df['gasto_pc_educacao'] = df['gasto_pc_educacao'].fillna(np.mean(df['gasto_pc_educacao']))

df['exp_anos_estudo'] = df['exp_anos_estudo'].fillna(np.mean(df['exp_anos_estudo']))

df['exp_vida'] = df['exp_vida'].fillna(np.mean(df['exp_vida']))

df['idhm'] = df['idhm'].fillna(np.mean(df['idhm']))

df['indice_governanca'] = df['indice_governanca'].fillna(np.mean(df['indice_governanca']))

# Passar densidade_dem para float

# df['densidade_dem'] = df['densidade_dem'].astype(float)

# ValueError: could not convert string to float: '1,973.60'

def convertaStringParaFloat(valor):

    if (type(valor) == str):

        valor = valor.replace(',','')

        valor = float(valor)

    return valor

df['densidade_dem'] = df['densidade_dem'].map(convertaStringParaFloat)

df['densidade_dem'] = df['densidade_dem'].fillna(np.mean(df['densidade_dem']))

df.info()

# Converta tipo Area para Float

df['area'] = df['area'].map(convertaStringParaFloat)
df['estado'] = df['estado'].astype('category')
df['estado'] = df['estado'].cat.codes
df['municipio'] = df['municipio'].astype('category')
df['municipio'] = df['municipio'].cat.codes
df.head()
#df.append(pd.get_dummies(df['regiao'], prefix='regiao'))

df = pd.concat([df, pd.get_dummies(df['regiao'], prefix='regiao').iloc[:, :-1]], axis =1)

del df['regiao']

df.info()
df = pd.concat([df, pd.get_dummies(df['porte'], prefix='porte').iloc[:, :-1]], axis =1)

del df['porte']

f = lambda x: x.replace('ID_ID_','')

df['codigo_mun'] = df['codigo_mun'].map(f)

df['codigo_mun'] = df['codigo_mun'].astype(int)

def transformePorcentagem(valorString):

    if (valorString == '#DIV/0!'):

        return 0

    valorSemSinal = valorString.replace('%', '')

    valor = int(valorSemSinal) / 100

    return valor
df['comissionados_por_servidor'] = df['comissionados_por_servidor'].map(transformePorcentagem)
def corrijaValoresAcimaDeCem(valor):

    if (valor > 100):

        print(valor)

        return 0

    else:

        return valor
df['comissionados_por_servidor'] = df['comissionados_por_servidor'].map(corrijaValoresAcimaDeCem)
df['comissionados_por_servidor'].head()
df['comissionados_por_servidor'] = df['comissionados_por_servidor'].fillna(np.mean(df['comissionados_por_servidor']))
def retiraSinalGrau(valor):

    if (type(valor) == str):

        valor = valor.replace('º', '')

        return int(valor)

    else:

        return 3343

# 3343 - É o último Grau.

        
df['ranking_igm'] = df['ranking_igm'].map(retiraSinalGrau)
df['ranking_igm'] = df['ranking_igm'].astype('category')
df['ranking_igm'] = df['ranking_igm'].cat.codes
boxplot = df.boxplot(column=['hab_p_medico'])
def retirarOutliers(valor):

    if (valor > 2174):

        return 2174

    else:

        return valor
df['hab_p_medico'] = df['hab_p_medico'].map(retirarOutliers)
boxplot = df.boxplot(column=['idhm'])
boxplot = df.boxplot(column=['perc_pop_econ_ativa'])
def retirarOutliers(valor):

    if (valor < 0.3):

        print(valor)

        return 0.3

    else:

        return valor
df['perc_pop_econ_ativa'] = df['perc_pop_econ_ativa'].map(retirarOutliers)
boxplot = df.boxplot(column=['anos_estudo_empreendedor'])
def limparAnosEstudos(valor):

    if (valor < 1.9):

        return 2

    elif (valor > 11):

        return 10

    else:

        return valor

    
df['anos_estudo_empreendedor'] = df['anos_estudo_empreendedor'].map(limparAnosEstudos)
boxplot = df.boxplot(column=['jornada_trabalho'])
def limparJornada(valor):

    if (valor > 50):

        return 50

    elif (valor < 30):

        return 30

    else:

        return valor
df['jornada_trabalho'] = df['jornada_trabalho'].map(limparJornada)
boxplot = df.boxplot(column=['exp_anos_estudo'])
def limparAnosEstudo(valor):

    if (valor > 11):

        return 11

    elif (valor < 7.5):

        return 7.5

    else:

        return valor
df['exp_anos_estudo'] = df['exp_anos_estudo'].map(limparAnosEstudo)
boxplot = df.boxplot(column=['participacao_transf_receita'])
def limparParticipacao(valor):

    if (valor < 50):

        return 55

    else:

        return valor
df['participacao_transf_receita'] = df['participacao_transf_receita'].map(limparParticipacao)
boxplot = df.boxplot(column=['nota_mat'])
def limparNotaMat(valor):

    if (valor > 6.32):

        return 6.32

    else:

        return valor
df['nota_mat'] = df['nota_mat'].map(limparNotaMat)
boxplot = df.boxplot(column=['populacao'])
np.percentile(df['populacao'], 75)  

np.percentile(df['populacao'], 25)  
irq = np.percentile(df['populacao'], 75) - np.percentile(df['populacao'], 25) 



df['populacao'] = irq
# df['populacao'] = df['populacao'].map(limparPopulacao)
boxplot = df.boxplot(column=['gasto_pc_saude'])
np.percentile(df['gasto_pc_saude'], 75)  

np.percentile(df['gasto_pc_saude'], 25)  
def limparGastoSaude(valor):

    if (valor > 561.1775):

        return 561.1775

    elif (valor < 385.015):

        return 385.015

    else:

        return valor
df['gasto_pc_saude'] = df['gasto_pc_saude'].map(limparGastoSaude)
boxplot = df.boxplot(column=['taxa_empreendedorismo'])
def limpaTaxaEmpreendedorismo(valor):

    if (valor > 0.4):

        return 0.4

    elif (valor < 0.12):

        return 0.12

    else:

        return valor
df['taxa_empreendedorismo'] = df['taxa_empreendedorismo'].map(limpaTaxaEmpreendedorismo)
boxplot = df.boxplot(column=['gasto_pc_educacao'])
def limpaGastoEducacao(valor):

    if (valor > 1100):

        return 1100

    elif (valor < 250):

        return 250

    else:

        return valor
df['gasto_pc_educacao'] = df['gasto_pc_educacao'].map(limpaGastoEducacao)
boxplot = df.boxplot(column=['area'])
irq = np.percentile(df['area'], 75)   - np.percentile(df['area'], 25)  

print(irq)
df['area'] = 1173.985
train_raw = df[~df['nota_mat'].isnull()]
train_raw.shape
for coluna in train_raw.columns:

    if (coluna == 'ranking_igm'):

        continue

    corr = np.corrcoef(train_raw['nota_mat'],train_raw[coluna])[0,1]

    if (corr < 0.4 and corr > -0.4):

        print(str(coluna) + ': '  + str(corr))

        

for coluna in train_raw.columns:

    if (coluna == 'ranking_igm'):

        continue

    corr = np.corrcoef(train_raw['nota_mat'],train_raw[coluna])[0,1]

    if (corr < -0.4):

        print(str(coluna) + ': '  + str(corr))
test = df[df['nota_mat'].isnull()]
test.shape
del test['nota_mat']
test.shape
from sklearn.model_selection import train_test_split
train, valid = train_test_split(train_raw, random_state=42)
removed_cols = ['nota_mat', 'Unnamed: 0', 'gasto_pc_saude']

#cols_test = ['idhm', 'exp_vida']
feats = [c for c in df.columns if c not in removed_cols]
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from xgboost import XGBClassifier
# 'XGB': XGBClassifier(n_estimators=200, learning_rate=0.09, random_state=42),



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
from sklearn.metrics import mean_squared_error
def run_model(model, train, valid, feats, y_name):

    model.fit(train[feats], train[y_name])

    preds = model.predict(valid[feats])

    return mean_squared_error(valid[y_name], preds)**(1/2)
train_feat = train[feats]
train_feat.head()
# Execução do Modelo

scores = []

for name, model in models.items():

    score = run_model(model, train, valid, feats, 'nota_mat')

    scores.append(score)

    print(name+':', score)
pd.Series(scores, index=models.keys()).sort_values(ascending=False).plot.barh()
rf = RandomForestRegressor(random_state=42)

rf.fit(train[feats], train['nota_mat'])

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
# gbm = GradientBoostingRegressor(random_state=42)

#alg = LinearRegression()

# https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

alg = GradientBoostingRegressor(random_state=42, min_samples_split = 500, min_samples_leaf = 50, max_depth = 8 , max_features = 'sqrt', subsample = 0.8)

# alg = GradientBoostingRegressor(random_state=42)

# alg = RandomForestRegressor(random_state=42, n_jobs=-1,n_estimators=30000,min_samples_leaf=650)
alg.fit(train[feats], train['nota_mat'])
train_preds = alg.predict(train[feats])
mean_squared_error(train['nota_mat'], train_preds)**(1/2)
valid_preds = alg.predict(valid[feats])
train.sample(10)
mean_squared_error(valid['nota_mat'], valid_preds)**(1/2)
test['nota_mat'] = np.exp(alg.predict(test[feats]))
test[['codigo_mun', 'nota_mat']].to_csv('Edison_Martins_13.csv', index=False)