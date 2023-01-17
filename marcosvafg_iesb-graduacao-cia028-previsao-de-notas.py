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
# Importando as bibliotecas de graficos

import seaborn as sns

import matplotlib.pyplot as plt
# Importando os dados

df = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-por.csv')



df.shape
# Visualizando os dados

df.head().T
# Analisando quantidades e tipos 

df.info()
# Analisando os dados



# Qtde de homens e mulheres

df['sex'].value_counts(normalize=True)
# Gráfico de homens e mulheres por escola

sns.catplot(x='school', hue='sex', data=df, kind='count')
# Correlação das variáveis numéricas

plt.figure(figsize= (15, 15))



sns.heatmap(df.corr(), square=True, annot=True, linewidth=0.5)
# Avaliando a variável target

sns.distplot(df['G3'], kde=True)
# Transformando os dados



# Transformando as variáveis binárias

df.loc[df['school'] == 'GP', 'school'] =  1

df.loc[df['school'] == 'MS', 'school'] =  0



df.loc[df['sex'] == 'M', 'sex'] =  1

df.loc[df['sex'] == 'F', 'sex'] =  0



df.loc[df['address'] == 'U', 'address'] =  1

df.loc[df['address'] == 'R', 'address'] =  0



df.loc[df['famsize'] == 'GT3', 'famsize'] =  1

df.loc[df['famsize'] == 'LE3', 'famsize'] =  0



df.loc[df['Pstatus'] == 'T', 'Pstatus'] =  1

df.loc[df['Pstatus'] == 'A', 'Pstatus'] =  0
# Transformando as variaveis yes/no

for col in ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',

            'internet', 'romantic']:

    df[col] = df[col].map({'yes': 1, 'no': 0})
# Transformando as variaveis categóricas usando o category do pandas

for col in ['Mjob', 'Fjob', 'reason', 'guardian']:

    # Transforma o texto em categoria

    df[col] = df[col].astype('category')

    # Salvando apenas os números de cada categoria

    df[col] = df[col].cat.codes
df.info()
# Vamos criar os datasets de treino e teste





# Importando o train_test_split do scikit learn

from sklearn.model_selection import train_test_split



# Separando o dataframe em treino e teste e usando uma semente aleatória

# para reproduzir os resultados

# Por padrão o test_size é 0.25

train, test = train_test_split(df, test_size=0.2, random_state=42)
train.shape, test.shape
# Selecionando as colunas que serão usadas para treino



# Não vamos usar as notas intermediárias para treinamento

# e também não usaremos a variável target 'G3'

remove = ['G1', 'G2', 'G3']



# Lista com as colunas a serem usadas

feats = [col for col in df.columns if col not in remove]
feats
# Criando o modelo de regressão

from sklearn import linear_model



regr = linear_model.LinearRegression()
# Treinando o modelo

regr.fit(X = train[feats], y = train['G3'])
# Gerando predições com base no modelo treinado

preds = regr.predict(X = test[feats])
preds
test['G3'].head()
# Avaliando o modelo de acordo com o Mean Squared Error

from sklearn.metrics import mean_squared_error



# Chamamos a função passando os valores reais e os valores preditos

mean_squared_error(test['G3'], preds)
# Avaliando o modelo por meio do r2 Score

from sklearn.metrics import r2_score



r2_score(test['G3'], preds)
# Qual o desempenho do modelo para os dados/variáveis de treino?



# Gerando previsões para os dados de treino usando o modelo treinado

preds2 = regr.predict(X = train[feats])



# Analisando as metricas

mean_squared_error(train['G3'], preds2), r2_score(train['G3'], preds2)
# Qual o desempenho do modelo usando as variáveis altamente correlacionadas?



# Selecionando as variaveis para treino

remove2 = ['G3']

feats2 = [col for col in train.columns if col not in remove2]



# Instanciar o modelo

regr2 = linear_model.LinearRegression()



# Treinar o modelo

regr2.fit(X = train[feats2], y = train['G3'])



# Prever usando o modelo

preds2 = regr2.predict(test[feats2])



# Analisar o desempenho

mean_squared_error(test['G3'], preds2), r2_score(test['G3'], preds2)
# Mudando de modelo



# Decision Tree



# Instanciando o modelo

from sklearn import tree

dc = tree.DecisionTreeRegressor(random_state=42)

# Treinar o modelo

dc.fit(train[feats], train['G3'])

# Fazendo previsões com o modelo

preds_dc = dc.predict(test[feats])

# Avaliando o modelo

mean_squared_error(test['G3'], preds_dc), r2_score(test['G3'], preds_dc)
# Mudando o modelo novamente



# Random Forest



# Instanciando o modelo

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=42)

# Treinar o modelo

rfr.fit(train[feats], train['G3'])

# Prever usando o modelo

preds_rfr = rfr.predict(test[feats])

# Avaliando o modelo

mean_squared_error(test['G3'], preds_rfr), r2_score(test['G3'], preds_rfr)