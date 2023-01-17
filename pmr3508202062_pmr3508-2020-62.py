import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import numpy as np
treino = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",na_values="?")

treino.head()
treino.isna().sum().sort_values(ascending=False)
def df_pc_50k(df):

    ricos = df[df['income']=='>50K'].count()[0]

    pc_df = ricos/df.shape[0]

    pc_df

    return pc_df



df_pc_50k(treino)
def pc_50k(df,coluna,tipo):

    a = df[df[coluna]==tipo]

    total_tipo = a.count()[0]

    total_rico = a[a['income']=='>50K'].count()[0]

    pc = total_rico/total_tipo

    return pc
treino['workclass'].value_counts()
for i in treino['workclass'].dropna().unique():

    print(i,pc_50k(treino,'workclass',i))
nanworkclass = treino[treino['workclass'].isna()]

df_pc_50k(nanworkclass)
treino['workclass']=treino['workclass'].fillna('Private')
for i in treino['workclass'].dropna().unique():

    print(i,pc_50k(treino,'workclass',i))
treino['occupation'].value_counts()
for i in treino['occupation'].dropna().unique():

    print(i,pc_50k(treino,'occupation',i))
nanoccupation = treino[treino['occupation'].isna()]

df_pc_50k(nanoccupation)
treino['occupation'] = treino['occupation'].fillna('Adm-clerical')
for i in treino['occupation'].dropna().unique():

    print(i,pc_50k(treino,'occupation',i))
treino['native.country'].value_counts()
for i in treino['native.country'].dropna().unique():

    print(i,pc_50k(treino,'native.country',i))
nannativecountry = treino[treino['native.country'].isna()]

df_pc_50k(nannativecountry)
treino['native.country']=treino['native.country'].fillna('United-States')
treino.isna().sum()
colunas = ['education.num','marital.status','relationship','race','sex']



for i in colunas:

    print(i,'\n')

    for j in treino[i].unique():

        print(j,pc_50k(treino,i,j))

    print('\n')
colunas.append('occupation')

colunas.append('native.country')

colunas
colunas.append('age')

colunas.append('capital.gain')

colunas.append('capital.loss')
teste = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",na_values="?")
teste.isna().sum()
teste['occupation']=teste['occupation'].fillna('Adm-clerical')

teste['native.country']=teste['native.country'].fillna('United-States')
teste.isna().sum()
from sklearn import preprocessing



Xtreino = treino[colunas]

Xtreino = Xtreino.apply(preprocessing.LabelEncoder().fit_transform)

Ytreino = treino['income']
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



acuracia = []



for k in [5,10,15,20,25,30,35,40]:

    knn = KNeighborsClassifier(n_neighbors= k)

    pontos = cross_val_score(knn, Xtreino, Ytreino, cv = 10)

    acuracia.append(np.mean(pontos))



best_K = (np.argmax(acuracia)+1)*5

print("Melhor Acuracia:",max(acuracia),"K =",best_K)
knn = KNeighborsClassifier(n_neighbors=best_K)

knn = knn.fit(Xtreino,Ytreino)
Xteste = teste[colunas]

Xteste = Xteste.apply(preprocessing.LabelEncoder().fit_transform)

Yteste = knn.predict(Xteste)
income = pd.DataFrame({'income':Yteste})

Id = teste['Id']

saida = pd.concat([Id,income],axis=1)
saida
saida.to_csv("submission.csv", index = False)