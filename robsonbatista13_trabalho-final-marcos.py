
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')
df.head()
df.info()
#Estatísticas Descritivas
df.describe()
# Verificando a distribuição da variável BAD(Target)
df['BAD'].plot.hist(bins=5)
#frenquência dos dados da variável BAD
df['BAD'].value_counts()
#Verificando a quantidade de valores missing nas variáveis
df.isnull().sum()
# Removendo os valores missing value da base
#df2 =df.copy()
df.dropna(axis=0,how='any',inplace= True)
df.info(), df.isna().any() 

#valores da variável target
df['BAD'].value_counts().plot(kind='bar',title='Frequência da Variável BAD')
df['BAD'].value_counts()
# Correlação das variáveis numéricas
plt.figure(figsize= (15, 8))
sns.heatmap(df.corr(), square=True, annot=True, linewidth=0.5)
#Distribuição da variável REASON por BAD
df.groupby(['BAD'])['REASON'].value_counts()
#Distribuição da variável JOB por BAD
df.groupby(['BAD'])['JOB'].value_counts()
# Gerando Dummies para modelos que utilizam apenas variaveis numéricas

df = pd.get_dummies(df, columns=['REASON', 'JOB'])
df.head().T
#Normalizando os dados

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#df = pd.DataFrame(sc.fit_transform(df), columns=df.columns)
# importando a biblioteca
from sklearn.model_selection import train_test_split

#Separando em treino e teste
treino, teste = train_test_split(df, test_size=0.20, random_state=42)

# Não vou usar o dataset de validação

treino.shape, teste.shape  
# Lista das colunas que serão usadas
usadas = [c for c in treino.columns if c not in ['BAD','REASON','JOB']]
#importando métrica
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

# importanto o modelo
from sklearn.ensemble import RandomForestClassifier

#instanciando o modelo
rf = RandomForestClassifier(n_estimators=200,random_state=42)
# treinando o modelo
rf.fit(treino[usadas], treino['BAD'])

# gerando predicoes do modelo com os dados de teste
pred_teste = rf.predict(teste[usadas])

#Medindo a acuracia nos dados de teste
accuracy_score(teste['BAD'],pred_teste), balanced_accuracy_score(teste['BAD'],pred_teste), f1_score(teste['BAD'],pred_teste)
# Avaliando a importancia de cada coluna (cada variável de entrada)
pd.Series(rf.feature_importances_, index=usadas).sort_values().plot.barh()
# importando a bilbioteca para plotar o gráfico de Matriz de Confusão
import scikitplot as skplt

# Matriz de Confusão - Dados de Validação
skplt.metrics.plot_confusion_matrix(teste['BAD'], pred_teste)
# Setando parametros
rf2 = RandomForestClassifier(max_depth=None, random_state=42, n_jobs=-1, n_estimators=500,
                            min_impurity_decrease=1e-3, min_samples_leaf=2,  class_weight='balanced')
# treinando o modelo RF2
rf2.fit(treino[usadas], treino['BAD'])
#relizando a predicao do RF2 com base teste
pred_teste2 = rf2.predict(teste[usadas])

#métrica para RF2 validacao
accuracy_score(teste['BAD'],pred_teste2), balanced_accuracy_score(teste['BAD'],pred_teste2), f1_score(teste['BAD'],pred_teste2)
# Matriz de Confusão - Dados de Validação
skplt.metrics.plot_confusion_matrix(teste['BAD'], pred_teste2)
# Importar o modelo
from xgboost import XGBClassifier

# Instanciar o modelo
xgb = XGBClassifier(n_estimators=900, n_jobs=-1, random_state=42, learning_rate=0.05)

# treinando o modelo
xgb.fit(treino[usadas],treino['BAD']) 


# Fazendo predições
pred_xgb_teste = xgb.predict(teste[usadas])

# Metrícas XGB teste
accuracy_score(teste['BAD'],pred_xgb_teste), balanced_accuracy_score(teste['BAD'],pred_xgb_teste), f1_score(teste['BAD'],pred_xgb_teste)
# Matriz de Confusão - Dados de Validação
skplt.metrics.plot_confusion_matrix(teste['BAD'], pred_xgb_teste)