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
import pandas as pd
import numpy as np
import holoviews as hv
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns

import holoviews as hv
hv.extension('bokeh', 'matplotlib', logo=False)
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import  warnings
warnings.filterwarnings("ignore")



#Carregando os dados
df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')
df.head()
# Verificando os tipos de dados
df.dtypes
#Verificando a quantidade de linhas e colunas
df.shape
#Análisando a base
df.info()
# Mostrando as colunas // ACHO QUE NÃO PRECISA COLOCA
df.columns
# Estatistica Descritiva. Vamos incluir TUDO....
df.describe(include='all')
numeric_feats = [c for c in df.columns if df[c].dtype != 'object' and c not in ['BAD']]
df_numeric_feats = df[numeric_feats]
# avaliando as variáveis numéricas
sns.pairplot(df_numeric_feats)
df_numeric_feats.hist(figsize=(20,8), bins=30)
# Analise Exploratória
df["BAD"].value_counts().plot.bar(title='BAD')
#Visualizando a variável categorica REASON
REASON_count= df["REASON"].value_counts().rename_axis('REASON').reset_index(name='Total Count')
df["REASON"].value_counts().plot.bar(title='REASON')
#visualizando  a variável categórica JOB
JOB_count= df["JOB"].value_counts().rename_axis('JOB').reset_index(name='Total Count')
df["JOB"].value_counts().plot.bar(title='JOB')
#Relação JOB vs BAD
JOB=pd.crosstab(df['JOB'],df['BAD'])
JOB.div(JOB.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, title='JOB vs BAD', figsize=(4,4))
#df = t.copy()
# Tratando colunas categóricas
#for col in df.select_dtypes(include='object').columns:
#    if df[col].isna().sum() > 0:
#         df[col].fillna(df[col].mode()[0], inplace=True)   
# Tratando colunas numéricas
#for col in df.select_dtypes(exclude='object').columns:
#    if df[col].isna().sum() > 0:
#        df[col].fillna(-1, inplace=True)      
#df.info()
def showBalance(df, col):
    for c in col:
        print('Distribuição da Coluna: ', c,'\n',df[c].value_counts(normalize=True),'\n')
    else:
       pass
        
showBalance(df, col=['REASON','JOB','BAD'])

# Finalmente, o número de linha de crédito aberta (CLNO) parece estatisticamente consistente em ambos os casos,
# sugerindo que essa variável não possui poder de discriminação significativo.
# Este método é primariamente baseado nas labels da colunas, porém podemos utilizar com um array booleano também. (Usando o loc)
# Uma informação importante sobre loc é: quando nenhum item é encontrado ele retorna um KeyError.
df.loc[df.BAD == 1, 'STATUS'] = 'DEFAULT'
df.loc[df.BAD == 0, 'STATUS'] = 'PAID'
# relação do empréstimo pagos. Pelo que mostra 81% dos empréstimos foram pagos.
#A discrepância de 4% observada não é estatisticamente significativa, dado o montante de empréstimos no conjunto de dados.
g = df.groupby('REASON')
g['STATUS'].value_counts(normalize=True).to_frame().style.format("{:.1%}")
# Matrix de correlação

corr = df.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(10,8))
#Generate Color Map
colormap = sns.diverging_palette(220, 10, as_cmap=True)
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
#Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
#show plot
plt.show()
#Verificando os missing 
df.isnull().sum()
# Criando uma cópia de um novo dataframe e vamos eliminar os NA, 
df2 = df.copy()
df2.dropna(axis=0,how='any',inplace= True)
df2.info(), df2.isna().any() 
df2.shape
# Analisando como ficou
df2.head()
# tranformando as colunas de object em categoria com codigos #
for col in df2.columns:
    if df2[col].dtype == 'object':
        df2[col]= df2[col].astype('category').cat.codes
df2.info()
# analisando REASON (MOTIVO)
df2['REASON'].value_counts()
#  Separando em Treino e Teste

treino, teste = train_test_split(df2, random_state=42)

#  Separando o treino e validacao, para refinar o modelo

#treino, validacao = train_test_split(treino, random_state=42)

treino.shape, teste.shape, #validacao.shape
# separar as colunas para usar no treino

usadas_treino = [c for c in treino.columns if c not in ['BAD','REASON','JOB','STATUS']]
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

rf = RandomForestClassifier(n_estimators=900, random_state=42)

rf.fit(treino[usadas_treino],treino['BAD'])
#Gerando as predições do modelo
rf_pred = rf.predict(teste[usadas_treino])

accuracy_score(teste['BAD'], rf_pred), f1_score(teste['BAD'],rf_pred)
#Olhando os valores da SITUACAO - TREINO

treino['BAD'].value_counts(normalize=True)
# Trabalhando com RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf2 = RandomForestClassifier(n_estimators=900, min_samples_split=5, max_depth=4, random_state=42)
rf2.fit(treino[usadas_treino],treino['BAD'])

#Gerando as predições do modelo
rf_pred2 = rf.predict(teste[usadas_treino])

accuracy_score(teste['BAD'], rf_pred2), f1_score(teste['BAD'],rf_pred2)
# Trabalhando com GBM
from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=42)
gbm.fit(treino[usadas_treino], treino['BAD'])

accuracy_score(validacao['BAD'], gbm.predict(validacao[usadas_treino]))
# Trabalhando com XGBoost
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=200, learning_rate=0.09, random_state=42)
xgb.fit(treino[usadas_treino], treino['BAD'])

#Gerando as predições do modelo
accuracy_score(teste['BAD'], xgb.predict(teste[usadas_treino]))
# Verificando e avaliando a importancia de cada coluna para o modelo RF

pd.Series(rf.feature_importances_, index=usadas_treino).sort_values().plot.barh()
# O modelo GBM em cada coluna...

pd.Series(gbm.feature_importances_, index=usadas_treino).sort_values().plot.barh()

# importando a bilbioteca para plotar o gráfico de Matriz de Confusão
import scikitplot as skplt

# Matriz de Confusão - Dados de Validação
skplt.metrics.plot_confusion_matrix(teste['BAD'], rf_pred)
#Verificando o desbalanceio da variável dependente
df['BAD'].value_counts()








# Dividindo o DataFrame
from sklearn.model_selection import train_test_split

# Treino e teste
treino, test = train_test_split(df, test_size=0.15, random_state=42)

# Veificando o tanho dos DataFrames
treino.shape, test.shape














