import pandas as pd

import sklearn

import sklearn.neighbors

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

import seaborn

import matplotlib.pyplot as plt

import numpy as np
dados = pd.read_csv('../input/adult-pmr3508/train_data.csv', na_values = '?')

dados.shape
dados.head()
dados.isnull().sum()
dados = dados.dropna()
dados.info()
dados = dados.replace(to_replace=['<=50K', '>50K'], value=[0, 1])

dados.head()
dados['income'].value_counts().plot(kind='pie',autopct='%.2f')
dados['education'].value_counts()
dados['education.num'].value_counts()
dados = dados.drop('education', axis = 1)

dados.head()
dados['native.country'].value_counts().plot(kind='bar')
other_countries = ['Mexico','Philippines','Germany','Puerto-Rico','Canada','El-Salvador','Cuba','India','England','Jamaica','South','Italy','China','Dominican-Republic','Vietnam','Guatemala','Japan','Columbia','Poland','Taiwan','Haiti','Iran','Portugal','Nicaragua','Peru','Greece','Ecuador','France','Ireland','Hong','Cambodia','Trinadad&Tobago','Laos','Thailand','Yugoslavia','Outlying-US(Guam-USVI-etc)','Hungary','Honduras','Scotland','Holand-Netherlands']

for i in other_countries:

                   dados = dados.replace(to_replace=['United-States',i], value=[0, 1])

dados['native.country'].value_counts().plot(kind='pie',autopct='%.2f')
dados.head()
print(dados['workclass'].value_counts().plot(kind='pie',autopct='%.2f'))
dados = dados.replace(to_replace=['Private','Self-emp-not-inc','Local-gov','State-gov','Self-emp-inc','Federal-gov','Without-pay'], value=[0, 1,2,3,4,5,6])

dados['workclass'].value_counts()
print(dados['marital.status'].value_counts().plot(kind='pie',figsize=(5,5),autopct='%.2f'))
dados = dados.replace(to_replace=['Married-civ-spouse','Never-married','Divorced','Separated','Widowed','Married-spouse-absent','Married-AF-spouse'], value=[0, 1,2,3,4,5,6])

dados.head()
dados['occupation'].value_counts().plot(kind='bar')
dados = dados.replace(to_replace=['Prof-specialty','Craft-repair','Exec-managerial','Adm-clerical','Sales','Other-service','Machine-op-inspct','Transport-moving','Handlers-cleaners','Farming-fishing','Tech-support','Protective-serv','Priv-house-serv','Armed-Forces'], value=[0, 1,2,3,4,5,6,7,8,9,10,11,12,13])

dados.head()
dados['sex'].value_counts().plot(kind ='pie',autopct='%.2f')
dados = dados.replace(to_replace=['Male','Female'], value=[0, 1])

dados.head()
dados['relationship'].value_counts().plot(kind ='pie',autopct='%.2f')
dados = dados.replace(to_replace=['Husband','Not-in-family','Own-child','Unmarried','Wife','Other-relative'], value=[0,1,2,3,4,5])

dados.head()
dados['race'].value_counts().plot(kind='pie',figsize = (5,5),autopct='%.2f')
dados = dados.replace(to_replace=['White','Black','Asian-Pac-Islander','Amer-Indian-Eskimo','Other'],value=[0,1,2,3,4])

dados.tail()
plt.figure(figsize=(11,11))

corte = np.triu(dados.corr())



seaborn.heatmap(dados.corr(), mask = corte, annot = True, fmt='.2g', vmax = 0.5, vmin = -0.5, center = 0, cmap = 'seismic')

plt.show()
colunas_a_tirar = ['workclass', 'fnlwgt','occupation','race','sex','capital.loss','native.country']

dados = dados.set_index('Id')

dados_prontos = dados

for i in colunas_a_tirar:

    dados_prontos = dados_prontos.drop(i, axis = 1)

dados_prontos.head()
X = dados_prontos.drop('income', axis = 1)

Y = dados_prontos['income']
acmax = 0

for k in range (1,31):

    classif = KNeighborsClassifier(n_neighbors = k)

    acuracia = cross_val_score(classif, X, Y, cv = 8)

    if acuracia.mean() > acmax:

        acmax = acuracia.mean()

        melhor_k = k

        print('Quando k = ', k, 'as acurácias são', acuracia,', a média das acurácias é', round(acuracia.mean(),4), 'e esse é o melhor resultado até agora')

    else: print('Quando k = ', k, 'as acurácias são', acuracia,'e a média das acurácias é', round(acuracia.mean(),3))

print('Após diversos testes, percebemos que o melhor k para o classificador é ', melhor_k, 'com uma média das suas acurácias de',acmax)
classif = KNeighborsClassifier(n_neighbors = melhor_k)
classif.fit(X,Y)
teste = pd.read_csv('../input/adult-pmr3508/test_data.csv')

colunas_a_tirar = ['Id','workclass', 'education','fnlwgt','occupation','race','sex','capital.loss','native.country']



for i in colunas_a_tirar:

    teste = teste.drop(i, axis = 1)



teste = teste.replace(to_replace=['Married-civ-spouse','Never-married','Divorced','Separated','Widowed','Married-spouse-absent','Married-AF-spouse'], value=[0, 1,2,3,4,5,6])

teste = teste.replace(to_replace=['Husband','Not-in-family','Own-child','Unmarried','Wife','Other-relative'], value=[0,1,2,3,4,5])

teste.head()
resultados = classif.predict(teste)
final = pd.DataFrame(data = resultados)

final = final.replace(to_replace=[0,1], value=['<=50K','>50K'])

final['income']=final[0]

final = final.drop(0,axis=1)

final.head()
final['income'].value_counts().plot(kind='pie',autopct='%.2f')
final.to_csv("submission.csv", index = True, index_label = 'Id')