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
# importando o dataset

df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')

df.head()
# Verificando as informações do dataset

df.shape, df.info()
# realizando análise exploratória



df.describe(include='all')
# Verificando o percentual de empréstimos não pagos (BAD = 1)



print(df['BAD'].value_counts())

print(df['BAD'].value_counts(normalize=True))
# Verificando estatísticas básicas dos valores de empréstimos. A média está em 18.607.



df['LOAN'].describe()
# Fazer tratamento dos Dados verificando a existência de missing values ou valores null



MissingValues =df.isnull().sum().rename_axis('Colunas').reset_index(name='Missing Values')

MissingValues
# utilizando a biblioteca Pandas Profiling para gerar um report com análise de todas os campos e suas principais estatísticas.



import pandas_profiling as pp



pp.ProfileReport(df)
# avaliação das variáveis numéricas por meio de histogramas

import matplotlib.pyplot as plt

%matplotlib inline

df.hist(figsize=(20,10))
# Criando um dataframe para empréstimos que não foram pagos



df_bad = df[df['BAD'] == 1]
# Verificando o Total dos empréstimos não pagos (20.120.400). Este total é relevante para estudar os potenciais maus pagadores.



df_bad['LOAN'].sum()
# Explorando as informações

df['DELINQ'].value_counts()
print(df['LOAN'].sum())

print(df[df['DELINQ'] != 0.0]['LOAN'].sum())

print(df[df['DELINQ'] == 0.0]['LOAN'].sum())
# Distribuição por profissão

%matplotlib inline



df['JOB'].value_counts().plot.bar()
# Distribuição por motivo do empréstimo

%matplotlib inline



df['REASON'].value_counts().plot.bar()
import matplotlib.pyplot as plt

import seaborn as sns



# Plotando a correlação



# Aumentando a area do grafico

f, ax = plt.subplots(figsize=(15,6))

sns.heatmap(df.corr(), annot=True, fmt='.2f', linecolor='red', ax=ax, lw=1)
# Identificando valores de hipoteta devido nulo.



df[df['MORTDUE'].isna()]
# Substituindo Nan por 0



df.fillna(0, inplace = True)
# criando uma nova base para proteger a base original



df_n = df
# Criando uma nova coluna onde 0 não é maior e 1 é maior que, para o campo da hipotéca ser maior que o valor da propriedade



df_n['HIP_M_PROP'] = df_n['VALUE'] - df_n['MORTDUE']



HIP_M_PROP = []
# determinar as categorias

for valor in df_n['HIP_M_PROP']:

    if valor <  0.0:

        HIP_M_PROP.append(1)

    elif valor >= 0.0:

        HIP_M_PROP.append(0)

        

df_n['HIP_M_PROP'] = HIP_M_PROP
# Criando nova coluna com valores 1 (sim) e 0 (não) para definir se o valor do empréstimo é maior que o valor da hipoteca



df_n['LOAN_M_HIP'] = df_n['LOAN'] - df_n['MORTDUE']



LOAN_M_HIP = []



# determinar as categorias

for valor in df_n['LOAN_M_HIP']:

    if valor <  0.0:

        LOAN_M_HIP.append(1)

    elif valor >= 0.0:

        LOAN_M_HIP.append(0)

        

df_n['LOAN_M_HIP'] = LOAN_M_HIP
features = df_n.columns.difference(['BAD','REASON','JOB'])



X = df_n[features].values

y = df_n['BAD'].values
# dividir a base em treino e teste



import pandas as pd



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(df_n.drop('BAD',

                                                    axis=1),

                                                    df_n['BAD'],

                                                    test_size=0.3,

                                                    random_state=42)
# criação das bases e seus tamanhos

x_train.shape
x_test.shape
y_train.shape
y_test.shape
# base de treino

x_train.head()
# Importando o método do scikitlearn para divisão do dataframe de treino em treino e validação



from sklearn.model_selection import train_test_split
# Dividindo a base de treino para teste



train, valid = train_test_split(x_train, random_state=42)
# base de validação

valid.head()
valid.shape
train.shape
# árvore de decisão

from sklearn.tree import DecisionTreeClassifier



classifier_dt = DecisionTreeClassifier(random_state=1986,

                           criterion='gini',

                           max_depth=3)

classifier_dt.fit(X, y)
# validar modelo

from sklearn.model_selection import cross_val_score



scores_dt = cross_val_score(classifier_dt, X, y,

                            scoring='accuracy', cv=6)



print(scores_dt.mean())
# predição com Ensemble



from sklearn.ensemble import RandomForestClassifier



classifier_rf = RandomForestClassifier(random_state=1986,

                           criterion='gini',

                           max_depth=10,

                           n_estimators=30,

                           n_jobs=-1)

scores_rf = cross_val_score(classifier_rf, X, y,

                            scoring='accuracy', cv=6)



print(scores_rf.mean())
# tentando uma nova modelagem e medindo a importância da features



classifier_rf.fit(X, y) 



features_importance = zip(classifier_rf.feature_importances_, features)

for importance, feature in sorted(features_importance, reverse=True):

    print("%s: %f%%" % (feature, importance*100))
# nova modelagem com Variável resposta: BAD e Variável explicativa: DEBTINC
# Identificando a frequência da variável DEBTINC



df['DEBTINC'].value_counts()
# Separando os dataframes onde o count é nulo



teste = df[df['DEBTINC'] == 0.0]



treino = df[df['DEBTINC'] != 0.0]
treino.shape
teste.shape
# Modelo RandomForest



# método do scikitlearn para divisão e instanciando o modelo



from sklearn.model_selection import train_test_split



rf = RandomForestClassifier(n_jobs=-1, n_estimators=200, oob_score=True, random_state=42)
# Removendo as colunas de resposta



removed_cols = ['BAD','DEBTINC','JOB','REASON']
# Criar a lista da colunas de entrada



feats = [c for c in train.columns if c not in removed_cols]
# Treinamento do modelo com as variáveis de entrada e as de resposta

rf.fit(treino[feats], treino['BAD'])
# Previsão da variável de teste usando o modelo treinado

teste['BAD'] = rf.predict(teste[feats]).astype(int)
from sklearn.ensemble import RandomForestClassifier



classifier_rf = RandomForestClassifier(random_state=1986,

                           criterion='gini',

                           max_depth=10,

                           n_estimators=30,

                           n_jobs=-1)

scores_rf = cross_val_score(classifier_rf, X, y,

                            scoring='accuracy', cv=6)



print(scores_rf.mean())
# Este modelo apresentou um desempenho melhor no percentual de predição.

# Há uma margem de diferença entre a base primária e a base testada de 10%
# Verificação das previsões

teste['BAD'].value_counts(normalize=True)
df['BAD'].value_counts(normalize=True)