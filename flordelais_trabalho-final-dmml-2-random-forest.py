# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Ler o arquivo e gravar no DataFrame
df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')
#Apresentar o cabeçalho e os 5 primeiros registros
df.head()
# Verificar quais são as colunas objetct
df.select_dtypes('object').head()
# Analisando os registros  da coluna REASON
df['REASON'].value_counts()
# Analisando os registros com os valores missing
pd.isnull(df).sum()
#Apresentar a quantidade de registros e colunas
df.shape
#Deletar os registros com dados nulos
df_all = (df.dropna(axis=0))
#Verificar se ainda existe registros nulos
pd.isnull(df_all).sum()
#Apresentar a quantidade de registros e colunas após a exclusão dos registros nulos
df_all.shape
# Transformar os registros 'DebtCon' em 1 e 'HomeImp' em 0
# Na coluna REASON
mapeamento = { 'DebtCon': 1, 'HomeImp': 0}

df_all['REASON'] = df_all['REASON'].replace(mapeamento).astype(int)

#Verificar a transformação da coluna 'REASON'
df_all.head()
# Analisando os dados da coluna JOB
df['JOB'].value_counts()
# Transformar 'Other' em 0, 'ProfExe' em 1, 'Office'em 2, 'Mgr'em 3, 'Self'em 4, 'Sales' em 5
# nas coluna JOB
mapeamento = { 'Other': 0, 'ProfExe': 1, 'Office': 2, 'Mgr': 3, 'Self': 4, 'Sales': 5}

df_all['JOB'] = df_all['JOB'].replace(mapeamento).astype(int)
#Verificar a transformação da coluna 'JOB'
df_all.head()
#Quantidade de registros por tipo de trabalho
df.groupby('JOB')['JOB'].count()
#Descrever os dados quantitativos
df_all.describe()
import seaborn as sns

sns.set_style('darkgrid')
sns.distplot(df['YOJ'])
# Definindo as colunas para treinamento
feats = [c for c in df_all.columns if c not in ['BAD']]
#Apresentar as colunas
feats
# importando a biblioteca para definir teste e treino

from sklearn.model_selection import train_test_split
# Separar os dataframes
train, test = train_test_split(df_all, random_state=42)


#Verificar a quantidade de registros de treino
print("Treino: ", train.shape)
print("Teste: ",test.shape)
# Instanciando o random forest classifier utilizando 200 árvores
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200)
# Treinando o modelo
rf.fit(train[feats], train['BAD'])
# Gerar as predicoes do modelo com os dados de teste
pred_test = rf.predict(test[feats])
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Apresentar a acurácia do modelo
print("Accuracy:",metrics.accuracy_score(test['BAD'], pred_test))

accuracy = metrics.accuracy_score(test['BAD'], pred_test)
# Apresentar a balanced do modelo
balanced = balanced_accuracy_score(test['BAD'],pred_test)
print("Balanced:",balanced)
# Apresentar a F1 score do modelo
score = f1_score(test['BAD'],pred_test)
print("f1 score:",score)
# Apresentar as colunas por importância
(pd.Series(rf.feature_importances_, index=feats)
   .nlargest(13)
   .plot(kind='barh')) 
#Importar as bibliotecas
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import scikitplot as skplt
#Apresentar a matriz de confusão
matriz = confusion_matrix(test['BAD'], pred_test)

matriz
# Apresentar o gráfico da matriz de confusão
skplt.metrics.plot_confusion_matrix(test['BAD'], pred_test)