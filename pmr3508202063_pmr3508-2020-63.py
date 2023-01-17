#importando as bibliotecas necessária
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
train_data = pd.read_csv("../input/adult-pmr3508/train_data.csv",na_values="?")
train_data = train_data.set_index('Id')
#visão geral da base de dados
train_data.head()
#lista com o nome das colunas para facilitar próximas analises
train_data.columns
#descrevendo a base para verificar anormalidades
train_data.describe()
#aqui podemos ver que temos 3 colunas com grandes quantidades de missing data
train_data.isnull().sum()

#verificando as colunas
print(train_data['workclass'].describe())
print()
print()
print(train_data['occupation'].describe())
print()
print()
print(train_data['native.country'].describe())
#preenchendo os missing data com a moda
moda1 = train_data['workclass'].mode()[0]
train_data['workclass'].fillna(moda1, inplace = True)

moda2 = train_data['occupation'].mode()[0]
train_data['occupation'].fillna(moda2,inplace = True)

moda3 = train_data['native.country'].mode()[0]
train_data['native.country'].fillna(moda3,inplace = True)
#missing data preenchida com sucesso
train_data.isnull().sum()

train_data.columns
#aqui vemos que o income esta relacionado com sexo
sns.countplot( x = "sex", data = train_data,palette = 'hls',hue = "income", )
# e também income esta relacionado com idade
sns.countplot( x = "age", data = train_data,palette = 'hls',hue = "income", saturation = 10)
#podemos ver que quem trabalha mais tem uma pequena tendencia a ganhar mais
sns.catplot(x = 'hours.per.week',y='income',data = train_data,kind = 'bar', palette = 'hls',hue = "income", saturation = 10)
#Primeiro vamos transformar as variaveis categoricas em nao categoricas
train_data.dtypes
obj = train_data.select_dtypes(['object']).columns
train_data[obj] = train_data[obj].astype('category')
train_data[obj]=train_data[obj].apply(lambda x: x.cat.codes)
train_data[obj]
train_data['income'].value_counts()
#analisando os paises vemos que temos uma maioria absoluta, para fins de precisão vamos substituir o 39 por 1 e o resto por 0
train_data['native.country'].value_counts()
def converte(x):
    if x == 39:
        x = 1
    else:
        x = 0
    return x
train_data['native.country'] = train_data['native.country'].apply(converte)
train_data['native.country'].value_counts()
train_data = train_data.drop(columns = 'fnlwgt')
#por ter versão numérica, podemos dropar education
train_data.drop(columns = ['education'],inplace = True)

train_data['education.num'].dtype
#Aqui vemos que nenhuma das colunas possuem grande correlações, podemos seguir com análise
sns.heatmap(train_data.corr())
train_data.head()
test_data = pd.read_csv("../input/adult-pmr3508/test_data.csv",na_values = "?")
test_data.set_index('Id',inplace = True)
#preenchendo os missing data com a moda
moda1 = test_data['workclass'].mode()[0]
test_data['workclass'].fillna(moda1, inplace = True)

moda2 = test_data['occupation'].mode()[0]
test_data['occupation'].fillna(moda2,inplace = True)

moda3 = test_data['native.country'].mode()[0]
test_data['native.country'].fillna(moda3,inplace = True)

obj = test_data.select_dtypes(['object']).columns
test_data[obj] = test_data[obj].astype('category')
test_data[obj]=test_data[obj].apply(lambda x: x.cat.codes)
test_data[obj]
test_data['native.country']
def converte2(x):
    if x == 38:
        x = 1
    else:
        x = 0
    return x
test_data['native.country'] = test_data['native.country'].apply(converte2)
test_data.drop(columns = ['education','fnlwgt'],inplace = True)
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_data.drop('income',axis = 1))
train_normalizado = scaler.transform(train_data.drop('income',axis = 1))
train = pd.DataFrame(train_normalizado,columns = train_data.columns[:-1])
train
scaler.fit(test_data)
test_normalizado = scaler.transform(test_data)
test = pd.DataFrame(test_normalizado,columns = test_data.columns)
test
x_train = train
y_train = train_data.income
'''%%time

acc = []

for k in range(15, 40):
    knn = KNeighborsClassifier(k, p = 1)
    scores = cross_val_score(knn, x_train, y_train, cv = 10)
    acc.append(np.mean(scores))

bestK = np.argmax(acc) + 15
print("Best acc: {}, K = {}".format(max(acc), bestK))'''
knn = KNeighborsClassifier(32, p = 1)
scores = cross_val_score(knn, x_train, y_train, cv = 10)
print(scores)
np.mean(scores)
knn.fit(x_train, y_train)
x_test = test
y_test = knn.predict(x_test)
#aqui ainda está binário, para ver basta contar os valores, o que tiver mais é income maior que 50k
y_test
#aqui podemos ver que está relativamente coerente com a base de treinos, e que o 0 correspondo a menos que 50k
y_test = pd.DataFrame(y_test)
y_test[0].value_counts()
#convertendo
def volta(x):
    if x == 0:
        x = '<=50K'
    else:
        x = '>50K'
    return x
        
prediction = y_test
prediction.columns = ['income']
prediction = prediction['income'].apply(volta)

prediction = pd.DataFrame(prediction)
prediction['Id'] = prediction.index
prediction = prediction[['Id','income']]
prediction.head()
prediction.to_csv('prediction.csv',index = False)
