# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/gas-prices-in-brazil/2004-2019.tsv',sep='\t')

df.head()
df.describe()
df.info()
df.drop(['Unnamed: 0','UNIDADE DE MEDIDA'],axis=1,inplace=True)
import matplotlib.pyplot as plt

import seaborn as sns
df.PRODUTO.unique().tolist()
df.groupby('PRODUTO')

df_etanol = df.loc[df.PRODUTO =='ETANOL HIDRATADO']

df_gasol = df.loc[df.PRODUTO =='GASOLINA COMUM']

df_glp = df.loc[df.PRODUTO =='GLP']

df_gnv = df.loc[df.PRODUTO == 'GNV']

df_diesel = df.loc[df.PRODUTO == 'ÓLEO DIESEL']

df_diesel_s10 = df.loc[df.PRODUTO == 'ÓLEO DIESEL S10']
numerics = ['MARGEM MÉDIA REVENDA','DESVIO PADRÃO DISTRIBUIÇÃO','PREÇO MÍNIMO DISTRIBUIÇÃO','PREÇO MÁXIMO DISTRIBUIÇÃO','COEF DE VARIAÇÃO DISTRIBUIÇÃO']
for i in numerics:

    df_gasol[i] = pd.to_numeric(df_gasol[i].str.replace('-',''))
df_gasol['PREÇO MÉDIO DISTRIBUIÇÃO'] = pd.to_numeric(df_gasol['PREÇO MÉDIO DISTRIBUIÇÃO'].str.replace('-',''))
sns.boxplot(data=df_gasol[['PREÇO MÉDIO REVENDA','PREÇO MÉDIO DISTRIBUIÇÃO']])

(df_gasol['PREÇO MÉDIO REVENDA']/ df_gasol['PREÇO MÉDIO DISTRIBUIÇÃO']).mean() -1

# O preço de revenda é 15,75% maior que o de distribuição em média
plt.figure(figsize=(10,8))

sns.lineplot(data=df_gasol,y='PREÇO MÉDIO REVENDA',x='MÊS',hue='REGIÃO')

#REGIÃO NORTE E CENTRO-OESTE POSSUEM GASOLINA MAIS CARA DEVIDO A DIFICULDADE DE DISTRIBUIÇÃO

#A GASOLINA CAI ENTRE ABRIL E AGOSTO (PERIODO DE BAIXA), ENTRETANTO ESSA TENDÊNCIA NÃO TEM ACONTECIDO NOS ULTIMOS ANOS
plt.figure(figsize=(10,8))

sns.lineplot(data=df_gasol,y='PREÇO MÉDIO REVENDA',x='ANO',hue='REGIÃO')

#CRESCIMENTO DE 2004 ATÉ O PRIMEIRO SEMESTRE DE 2019
plt.figure(figsize=(20,8))

sns.barplot(data=df_gasol,y='PREÇO MÉDIO REVENDA',x='ESTADO')

plt.xticks(fontsize=8,rotation=35)

#ANALISANDO COMO APESAR DO PREÇO MÉDIO DE REVENDA SER MENOR DURANTE MAIOR PARTE DO TEMPO ANALISADO

#O RIO SEMPRE TEVE UMA DAS GASOLINAS MAIS CARAS DO PAÍS
df_gasol.drop(['DATA INICIAL','DATA FINAL'],axis=1,inplace=True)

#REDUNDANTE COM 'ANO' E 'MÊS'
df_gasol.head(8)
df_gasol.info()
fig= plt.figure(figsize=(13,10))

sns.heatmap(df_gasol.corr(),annot=True,cmap='PuBuGn')
df_gasol1= df_gasol[['ANO','MÊS']]

#ÚNICAS FEATURES POSSIVEIS DE PREDIZER VALORES REAIS
from sklearn.model_selection import train_test_split, GridSearchCV

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error,mean_squared_log_error,r2_score,explained_variance_score
y = df_gasol['PREÇO MÉDIO REVENDA']
x_treino,x_teste,y_treino,y_teste = train_test_split(df_gasol1,y,test_size=0.15,random_state=23)
xgbr= XGBRegressor()

xgbr.fit(x_treino,y_treino)
xgbr.score(x_teste,y_teste)
xgbr1= XGBRegressor()

parameters = {'objective':['reg:linear'],

              'learning_rate': [.1, 0.2, .5,0.01],

              'max_depth': [2,3],

              'min_child_weight': [0.5,1],

              'subsample': [0.5,1],

              'colsample_bytree': [0.5,1],

              'n_estimators': [100,200,300]}

xgb_grid = GridSearchCV(xgbr1,

                        parameters,

                        cv = 2,

                        n_jobs = 5,

                        verbose=True)

xgb_grid.fit(x_treino,y_treino)
xgb_grid.best_params_
improved_xgbr =XGBRegressor(colsample_bytree= 1, learning_rate= 0.41, max_depth= 3,

                          min_child_weight = 0.5, n_estimators = 333, objective = 'reg:linear', subsample=1)

improved_xgbr.fit(x_treino,y_treino)
improved_pred = improved_xgbr.predict(x_teste)

improved_xgbr.score(x_teste,y_teste)

print(mean_squared_error(y_teste,improved_pred))

print(mean_squared_log_error(y_teste,improved_pred))

print(r2_score(y_teste,improved_pred))

print(explained_variance_score(y_teste,improved_pred))
teste= [(2019,6),(2019,7),(2019,8),(2019,9),(2019,10),(2019,11),(2019,12)]

dft = pd.DataFrame(teste,columns=['ANO','MÊS'])

improved_xgbr.predict(dft)
df_gasol.loc[df_gasol.ANO == 2019].loc[df_gasol.MÊS == 6]['PREÇO MÉDIO REVENDA'].mean()

#Para Junho, o algoritmo previu 95,56% do valor