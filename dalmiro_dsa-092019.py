# Obter os dados

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

%matplotlib inline

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dsTreino_Y = pd.read_csv(dirname+'/y_treino.csv')

dsTreino_X = pd.read_csv(dirname+'/X_treino.csv')

dsTeste_X = pd.read_csv(dirname+'/X_teste.csv')

##Cria um ds para representar o cadastro de superfícies e gerar um id para relacionar com as outras tabelas

indice=0

dsSurface=pd.DataFrame(columns=['id_surface','surface'])

for i in dsTreino_Y.surface.unique():

    indice+= 1

    dsSurface = dsSurface.append({'id_surface': int(indice), 'surface':i},ignore_index=True)

dsSurface['id_surface'] = dsSurface['id_surface'].astype(int)

##Fazendo o relacionamento das tabelas 

dsSurface_ocorrencia = dsTreino_Y.merge(dsSurface,on = 'surface')

dsTreino = dsTreino_X.merge(dsSurface_ocorrencia,on = 'series_id')

dsTeste = dsTeste_X.merge(dsSurface_ocorrencia,on = 'series_id')

print('dsSurface_ocorrencia')

print(dsSurface_ocorrencia)

print('dsTreino')

print(dsTreino)

print('dsTeste')

print(dsTeste)
print('Treino')

print(dsTreino.isnull().any())

print('Teste')

print(dsTeste.isnull().any())

dsTreino.info()

dsTreino.describe()

## As 2 colunas estão como object, temos que tratar, ignorar ou deixar como nr.

## row_id acredito que poderemos ignorar, mas surface é nosso target e precisamos transformar em nr.

dsTreino.loc[:, ['row_id', 'surface']] 

##retiranto as colunas 

##dsTreino = dsTreino[dsTreino.columns.drop(['row_id'])]

#dsTreino.drop('row_id','surface, axis=1, inplace = True)

#dsTeste.drop('row_id', axis=1, inplace = True)

cols = ['row_id','surface']

dsTreino.drop(columns=cols, axis=1, inplace = True)

dsTeste.drop(columns=cols, axis=1, inplace = True)

dsTreino.info()
X = dsTreino.drop(columns=['id_surface'])

y = dsTreino['id_surface']
# Análise dos dados

columns=dsTreino.columns[:13]

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in zip(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    dsTreino[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()

dsTreino['id_surface'].value_counts().plot(kind='bar', figsize=(6,6))

plt.title('Tipos de piso')

plt.xlabel('Piso')

plt.ylabel('Frequência')

plt.show()
dsTreino.id_surface.value_counts()
from sklearn.model_selection import train_test_split

train,test=train_test_split(dsTreino,test_size=0.20,random_state=437,stratify=dsTreino['id_surface'])# stratify the outcome



X_train=train[train.columns[:13]]

X_test=test[test.columns[:13]]

y_train=train['id_surface']

y_test=test['id_surface']
from sklearn.tree import DecisionTreeClassifier

random_state=234

dtree = DecisionTreeClassifier(random_state=998)
dtree.fit(X_train,y_train)
from sklearn import metrics

pred = dtree.predict(X_test)

print("Acurácia",metrics.accuracy_score(pred,y_test))
df_submission = pd.DataFrame()

df_submission['series_id'] = y_test.index

df_submission['id_surface'] = y_test.values

dsSub = df_submission.merge(dsSurface,on = 'id_surface')
dsSubFinal =dsSub

dsSubFinal.drop(['id_surface'], axis=1, inplace=True)

dsSubFinal.to_csv(dirname+'submission_dsa092019.csv', index=False)

##pd.read_csv(dirname+'/y_treino.csv')
dsSubFinal.to_csv('submission_dsa092019.csv', index=False)