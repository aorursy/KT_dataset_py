import math # Matematica

import pandas as pd # Pre-processamento

import numpy as np # Algebra linear

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import sklearn.metrics as metrics

from scipy import stats

from scipy.stats import norm, skew

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn # ignora alguns avisos irritantes(das bibliotescas sklearn e seaborn)
# Lendo a base de dados

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train
test
sns.distplot(train['SalePrice'] , fit=norm);



# Obtem os paramentros ajustados pela função

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} e sigma = {:.2f}\n'.format(mu, sigma))



# Exibe a distribuição atual

plt.legend(['Normal dist. ($\mu=$ {:.2f} e $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequencia')

plt.title('Distribuição do Preço de Venda')



# Também obtem o QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
# Normaliza

train["SalePrice"] = np.log1p(train["SalePrice"])
sns.distplot(train['SalePrice'] , fit=norm);



# Obtem os paramentros ajustados pela função

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



# Exibe a distribuição atual

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequencia')

plt.title('Distribuição do Preço de Venda')



# Também obtem o QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
# Concatena as bases de dados para facilitar o pré-processamento

data = pd.concat([train, test])
data.set_index('Id', inplace = True)
data.isnull().sum()[:50]
data.info()
data['PoolQC'].fillna('None', inplace=True) # Valor NaN significa 'Sem Pscina'

data['MiscFeature'].fillna('None', inplace=True) # Valore NaN significa 'Sem Funcionalidade diversa'

data['Alley'].fillna('None', inplace=True) # Valor NaN significa 'Sem acesso ao beco'

data['Fence'].fillna('None', inplace=True) # Valor NaN significa 'Sem cerca'

data['FireplaceQu'].fillna('None', inplace=True) # Valor NaN significa 'Sem Lareira'
# Como a área de cada rua conectada à propriedade da casa, muito provavelmente tem uma área semelhante a outras casas em seu bairro, podemos preencher os valores que faltam pela mediana da 'LotFrontage' do bairro.

data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
# GarageType, GarageFinish, GarageQual e GarageCond: Repondo dados faltantes por None

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    data[col] = data[col].fillna('None')
# GarageYrBlt, GarageArea e GarageCars : Repondo dados faltantes por 0 (Pois Sem garagem = sem carros na garagem.)

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    data[col] = data[col].fillna(0)
# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath e BsmtHalfBath : Dados faltantes provavelmente são zeros (pois o lugar pode não ter porão)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    data[col] = data[col].fillna(0)
# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 e BsmtFinType2 : Para todas essas categorias que tem relação com porão, NaN ssignifica que pode não ter porão.

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    data[col] = data[col].fillna('None')
# MasVnrArea e MasVnrType : Provavelmente Na significa 'sem folheado de alvenaria'. Nos podemos colocar a área como 0 e o tipo como None. 

data["MasVnrType"] = data["MasVnrType"].fillna("None")

data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
# MSZoning (Classificação geral de zoneamento) : 'RL' é de longe o valor mais comum. Nesse caso nos podemos substituir  Na por 'RL'

data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])
# Utilities : Para essa caracteristica categorica todos os valores são "AllPub", exceto um "NoSeWa" e 2 NA. Como a casa com 'NoSewa' está no conjunto de treinamento, esse recurso não ajudará na modelagem preditiva. Podemos então removê-lo tranquilamente.

data = data.drop(['Utilities'], axis=1)
# Functional : A descrição dos dados diz que NA significa typical

data["Functional"] = data["Functional"].fillna("Typ")
# Electrical : Esse tem apenas um valor Na. Como a caracteristica mais comum é 'SBrkr', podemos substituir o valor faltante por este.

data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
# KitchenQual: Apenas um valor Na. Como a caracteristica mais comum é 'TA', podemos substituir o valor faltante por este.

data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
# Exterior1st e Exterior2nd : Denovo, como Exterior 1 e 2 tem apenas um valor faltante. Vamos substituir esse valor pelo valor mais comum da coluna.

data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])

data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
# SaleType : Preenchendo denovo com o valor mais comum: "WD"

data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
# MSSubClass : Na provavelmente significa Sem aula de construção. Vamos substituilos por None

data['MSSubClass'] = data['MSSubClass'].fillna("None")
data.info()
# Transformando algumas variáveis numericas que na verdade são categoricas



# MSSubClass=Sem aula de construção

data['MSSubClass'] = data['MSSubClass'].apply(str)





# Mudando OverallCond para variavel categorica

data['OverallCond'] = data['OverallCond'].astype(str)





# Ano e mes vendido transformados para variáveis categoricas

data['YrSold'] = data['YrSold'].astype(str)

data['MoSold'] = data['MoSold'].astype(str)
# Aplicando Label Encoding em algumas variáveis categóricas que podem conter informações em seu conjunto de ordenação



from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# Processa as colunas, e aplica LabelEncoder para variáveis categoricas

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(data[c].values)) 

    data[c] = lbl.transform(list(data[c].values))



# Formato        

print('Shape data: {}'.format(data.shape))
# Representa as variáveis categoricas de uma forma numérica

data = pd.get_dummies(data)

print(data.shape)
train = data[:1460]

test = data[1460:].drop('SalePrice', axis = 1)
train.shape, test.shape
X = train.drop('SalePrice', axis = 1)

y = train['SalePrice']
# Divide a base em treino e validação

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
xgb = XGBRegressor(booster='gbtree', colsample_bylevel=1,

                   colsample_bynode=1,colsample_bytree=0.6,

                   gamma=0, importance_type='gain',

                   learning_rate=0.01, max_delta_step=0,

                   max_depth=4,min_child_weight=1.5,

                   n_estimators=2500, n_jobs=1, nthread=None,

                   objective='reg:linear', reg_alpha=0.4640,

                   reg_lambda=0.6, scale_pos_weight=1,

                   silent=None, subsample=0.8, verbosity=1)
lgbm = LGBMRegressor(objective='regression',

                    num_leaves=4,

                    learning_rate=0.01,

                    n_estimators=11000,

                    max_bin=200,

                    bagging_fraction=0.75,

                    bagging_freq=5,

                    bagging_seed=7,

                    feature_fraction=0.4)
gboost = GradientBoostingRegressor(n_estimators=3000, 

                                   learning_rate=0.05,

                                   max_depth=4, 

                                   max_features='sqrt',

                                   min_samples_leaf=15, 

                                   min_samples_split=10, 

                                   loss='huber', 

                                   random_state =5)
xgb.fit(X_train, y_train)

lgbm.fit(X_train, y_train, eval_metric='rmsle')

gboost.fit(X_train, y_train)
pred2 = xgb.predict(X_test)

pred3 = lgbm.predict(X_test)

pred4 = gboost.predict(X_test)
print('Erro médio logarítmico de raiz quadrado test (XGB) = ' + str(math.sqrt(metrics.mean_squared_log_error(y_test, pred2))))

print('Erro médio logarítmico de raiz quadrado test (LGBM) = ' + str(math.sqrt(metrics.mean_squared_log_error(y_test, pred3))))

print('Erro médio logarítmico de raiz quadrado test (GBoost) = ' + str(math.sqrt(metrics.mean_squared_log_error(y_test, pred4))))
lgbm.fit(X, y)   # 0.12269 

xgb.fit(X ,y)    # 0.12495

gboost.fit(X, y) # 0.12333
prediction_lgbm =  np.expm1(lgbm.predict(test))

prediction_xgb = np.expm1(xgb.predict(test))

prediction_gboost = np.expm1(gboost.predict(test))
"""

prediction = ( prediction_lgbm * 0.38 + prediction_gboost * 0.35 + prediction_xgb * 0.27)   # 0.12006

prediction = ( prediction_lgbm * 0.4 + prediction_gboost * 0.35 + prediction_xgb * 0.25)    # 0.12007

prediction = ( prediction_lgbm * 0.45 + prediction_gboost * 0.35 + prediction_xgb * 0.2)    # 0.12012

prediction = ( prediction_lgbm * 0.55 + prediction_gboost * 0.45)                           # 0.12061

prediction = ( prediction_lgbm * 0.45 + prediction_gboost * 0.55)                           # 0.12069

prediction = ( prediction_gboost * 0.15 + prediction_lgbm * 0.7 + prediction_gboost * 0.15) # 0.12086

prediction = ( prediction_gboost * 0.2 + prediction_lgbm * 0.5 + prediction_gboost * 0.3)   # 0.12154

prediction = ( prediction_lgbm * 0.55 + prediction_xgb * 0.45)                              # 0.12155

"""
prediction = ( prediction_lgbm * 0.38 + prediction_gboost * 0.36 + prediction_xgb * 0.26)   # 0.12006
# Tranforma o resultado em um DataFrame

submission = pd.DataFrame({"Id": test.index,"SalePrice": prediction})
# Transforma o DataFrame em um arquivo csv para ser submetido no kaggle.

submission.to_csv('submission.csv', index=False)