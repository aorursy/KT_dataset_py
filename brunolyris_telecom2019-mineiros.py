

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Carregando os dados

df = pd.read_csv('/kaggle/input/telecom2019/database_telecom.csv', sep=';')



df.head()
# Importando bliotecas de plotagem e de regressão

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
# Verificando classe das variaveis

df.info()
# Resumo estatistico das variaveis.

df.describe()
# Visualizando a quantidade de  instalações por unidade federativa via grafico de barras

df.groupby('UF')['QTDE_INSTALACAO'].nunique().plot(kind='bar')

plt.show()



# Tabela de resumo das volumetrias por regional

df_agrupado=df.groupby('REGIONAL').sum()

df_agrupado
# Visualizando a quantidade de  instalações por Regional via grafico de barras

df.groupby('REGIONAL')['QTDE_INSTALACAO'].sum().plot(kind='bar')

plt.show()
# Histograma para visualização da distribuição de reclamações em ate 15 dias apos a instalaçao.

sns.distplot(df['QTDE_RECENTE'], bins = 15)
# Calculo de porcentagem de da quantidade de reclamações em ate 15 dias em relação ao volume total de instalações.

porc_reg=df_agrupado['QTDE_RECENTE'] /df_agrupado['QTDE_INSTALACAO']*100

porc_reg
# Visualização do resultado em grafio de barras - temos que a centro oeste apresenta ser a maior quantidade de 

# Reclamações em ate 15 dias em relação ao volume de cliente instalados

porc_reg.plot.bar()
# Abrindo a visão da regional centro oeste

df_agrupadoCO= df[df['REGIONAL']=='CENTRO OESTE']

df_agrupadoCO
# Iniciando base para analise preditiva com apenas os campo de quantidade de instalação e quantidade de recentes

base = df.drop(['CIDADE',

                'REGIONAL',

                'UF',

                'QTDE_DADOS',

                'QTDE_VOZ',

                'QTDE_TV',

                'CLT_B2C',

                'CLT_B2B',

               ],axis =1)

base
# Resumo por cidade que compoe a regiao centro oeste.

df_co=df_agrupadoCO.groupby('CIDADE').sum()

df_co
# Correlaçao para regressao linear

X = base.iloc[:,0].values

Y = base.iloc[:,1].values

correlacao = np.corrcoef(X,Y)

correlacao
# Modelo de regressao 

X = X.reshape(-1,1)

modelo = LinearRegression()

modelo.fit(X,Y)
# Obtendo o intecepto

modelo.intercept_
# Obtendo o coeficiente de correlação

modelo.coef_

# Traçando grafico de regressão

plt.scatter(X,Y)

plt.plot(X, modelo.predict(X), color = 'red')
# Previsao de 4230 instalações em Mineiros

modelo.intercept_ + modelo.coef_*4230
