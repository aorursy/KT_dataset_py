# Imports

# manipulação de dados

import pandas as pd

import numpy as np

# gráficos

import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt

# eliminação recursiva de atributos

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

# Preprocessamento

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

# Cross Validation

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

# Modelo XGBoost

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# evitar avisos de warnings

import warnings

warnings.filterwarnings("ignore")
# Importando dados de treino

treino = pd.read_csv("../input/santander-customer-satisfaction/train.csv")

teste = pd.read_csv("../input/santander-customer-satisfaction/test.csv")
# Observando as primeiras linhas

treino.head()
# Tipo dos dados

treino.dtypes
# Tamanho do banco de dados treino

len(treino)
# Resumo dos dados de treino

treino.describe()
# Verificando a proporção da variável TARGET

# 0 = Clientes satisfeitos e 1 = Clientes insatisfeitos

df = pd.DataFrame(treino.TARGET.value_counts())

df['Prop'] = 100 * df['TARGET'] / treino.shape[0]

df['Prop'].plot(kind = 'bar', title = 'Proporção (Target)', color = ['#1F77B4', '#FF7F0E']);
# É possível observar o valor -999999, o que caracteriza um valor missing

treino.var3.value_counts()
# Substituindo os -999999 por 2 e inserindo ao banco de dados

var3_1 = treino.var3.replace(-999999, 2)

treino.insert(2, 'var3_1', var3_1)
treino.var3_1.value_counts()
treino.describe()
# Histograma da variável var15 - possivelmente é a idade de cada cliente

sns.distplot(treino.var15, fit = stats.norm);
# Boxplot

sns.boxplot(x = "TARGET", y = "var15", data = treino);
# variável TARGET pela var15

sns.stripplot(x = "TARGET", y = "var15", data = treino, jitter = True);
# Contar Classes

conte_classe_0, conte_classe_1 = treino.TARGET.value_counts()



# Dividindo por classe

df_classe_0 = treino[treino['TARGET'] == 0]

df_classe_1 = treino[treino['TARGET'] == 1]

df_classe_0_UnderS = df_classe_0.sample(conte_classe_1)

df_treino_UnderS = pd.concat([df_classe_0_UnderS, df_classe_1], axis = 0)



# Mostrando como ficou a contagem da variável TARGET

print('Under Sampling Aleatório:')

print(df_treino_UnderS.TARGET.value_counts())

df_treino_UnderS.TARGET.value_counts().plot(kind = 'bar', title = 'Contagem (target)', color = ['#1F77B4', '#FF7F0E']);
# Separando a variável TARGET

array = df_treino_UnderS.values



X = array[:,1:371]

Y = array[:,371]



# Semente



seed = 123



# Criação do modelo

modelo = LogisticRegression(random_state = seed)



# RFE - Eliminação Recursiva de Atributos

rfe = RFE(modelo, 10) # Os 10 mais importantes

fit = rfe.fit(X, Y)



# Print dos resultados

print('Variáveis Preditoras:', treino.columns[1:371])

print('Variáveis Selecionadas: %s' % fit.support_)

# o número 1 são as variáveis que apresentaram melhor resultado

print('Ranking dos Atributos: %s' % fit.ranking_)

print('Número de Melhores Atributos: %d' % fit.n_features_)
# Variáveis selecionadas



var_selec_treino = df_treino_UnderS[['ind_var30', 'ind_var30_0','num_var5','num_var8_0','num_var13_0', 'num_var13_corto_0', 'num_var13_corto',

                              'num_var42', 'num_meses_var5_ult3', 'num_meses_var8_ult3']]
# Transformando os dados para a mesma escala (entre 0 e 1)

# Dados

X_treino = var_selec_treino



# Gerando a nova escala (normalizando os dados entre 0 e 1)

X_treino = MinMaxScaler().fit_transform(X_treino)



# Padronizando os dados

X_treino = StandardScaler().fit_transform(X_treino)
# Carregando os dados



Y_treino = df_treino_UnderS['TARGET'].values



# Definindo os valores para o número de folds

num_folds = 10

seed = 7



# Separando os dados em folds

kfold = KFold(num_folds, True, random_state = seed)



# Criando o modelo

modelo = XGBClassifier().fit(X_treino, Y_treino)



# Cross Validation

resultado = cross_val_score(modelo, X_treino, Y_treino, cv = kfold)
#Variáveis de teste selecionadas



var_selec_teste = teste[['ind_var30', 'ind_var30_0','num_var5','num_var8_0','num_var13_0', 'num_var13_corto_0', 'num_var13_corto',

                              'num_var42', 'num_meses_var5_ult3', 'num_meses_var8_ult3']]
# Dados de teste

X_test = var_selec_teste



# Normalizando os dados

X_test = MinMaxScaler().fit_transform(X_test)



# Padronizando os dados

X_test = StandardScaler().fit_transform(X_test)



# Previsões com os dados de teste

pred = modelo.predict(X_test)
# Dados de submissão

submission = pd.DataFrame()

submission['ID'] = teste["ID"]

submission['TARGET'] = pred



submission.to_csv('submission.csv', index = False)
print(submission)