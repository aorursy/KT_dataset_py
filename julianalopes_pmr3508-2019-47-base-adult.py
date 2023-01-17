import pandas as pd

import numpy as np

import math

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score
train_data = pd.read_csv('../input/adult-pmr3508/train_data.csv')
test_data= pd.read_csv('../input/adult-pmr3508/test_data.csv')
len(train_data)
train_data.head()
train_data.describe()
train_data.info()

#Transformar colunas categoricas em numericas, testando em workclass

label_workclass= train_data['workclass'].astype('category').cat.categories.tolist()
label_workclass
# Mapa para a reposição da coluna workclass

replace_map_workclass = {'workclass':{k: v for k,v in zip(label_workclass,list(range(1,len(label_workclass)+1)))}}
replace_map_workclass
# copiar para substituir com seguranca

train_data_replace = train_data.copy()
# substituir valores categoricos por numericos

train_data_replace.replace(replace_map_workclass, inplace = True)
train_data_replace.head()
train_data_replace = train_data_replace.drop(columns = 'education')
# o Mesmo para as colunas restantes

colunas_categoricas=['marital.status','occupation','relationship','race','sex', 'native.country']

replace_map=[0,0,0,0,0,0]

labels=[0,0,0,0,0,0]

i=0

for column in colunas_categoricas:

    labels[i] = train_data_replace[column].astype('category').cat.categories.tolist()

    replace_map[i]=  {column:{k: v for k,v in zip(labels[i],list(range(1,len(labels[i])+1)))}}

    i=i+1
for j in range(len(colunas_categoricas)):

    train_data_replace.replace(replace_map[j], inplace=True)
train_data_replace.head()

#dividindo em X e Y

X_train_data_replace = train_data_replace.iloc[:, 0:14]

Y_train_data_replace = train_data_replace.iloc[:,14]
# primeiro teste para k =3

classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(X_train_data_replace, Y_train_data_replace)
#realizando a previsão

y_pred = classifier.predict(X_train_data_replace)
print(y_pred)
#utilizando accuracia, poremo enunciado pede validacao cruzada

print(accuracy_score(Y_train_data_replace, y_pred))
#Descobrindo o melhor valor de k 

# iterar de k=3 ate k=50 



lista=[]

lista_score=[]

for k in range(3,50):

    classifier = KNeighborsClassifier(n_neighbors = k)

    classifier.fit(X_train_data_replace, Y_train_data_replace)

    y_pred = classifier.predict(X_train_data_replace)

    score = cross_val_score(classifier, X_train_data_replace, Y_train_data_replace, cv=5)

    med_score = np.mean(score)

    lista.append(k)

    lista_score.append(med_score)



plt.plot(lista,lista_score)
# Mudaria se tentassemos apenas os valore numericos?

numerics = ['int64']

X_train_data_numeric = train_data.select_dtypes(include=numerics)
X_train_data_numeric
#Descobrindo o melhor valor de k novamente

# iterar de k=3 ate k=181 (raiz quadrada do total de dados)





classifier = KNeighborsClassifier(n_neighbors = 18)

classifier.fit(X_train_data_numeric, Y_train_data_replace)

y_pred = classifier.predict(X_train_data_numeric)

score = cross_val_score(classifier, X_train_data_replace, Y_train_data_replace, cv=5)

med_score = np.mean(score)



print(med_score)
# repetindo para a base de teste

test_data_replace = test_data.copy()

colunas_categoricas=['workclass','marital.status','occupation','relationship','race','sex', 'native.country']

replace_map=[0,0,0,0,0,0,0]

labels=[0,0,0,0,0,0,0]

i=0

for column in colunas_categoricas:

    labels[i] = test_data_replace[column].astype('category').cat.categories.tolist()

    replace_map[i]=  {column:{k: v for k,v in zip(labels[i],list(range(1,len(labels[i])+1)))}}

    i=i+1



for j in range(len(colunas_categoricas)):

    test_data_replace.replace(replace_map[j], inplace=True)
test_data_replace.head()

X_test_data_replace = test_data_replace.drop(columns = 'education')
X_test_data_replace.head()
classifier.fit(X_train_data_replace, Y_train_data_replace)

y_pred = classifier.predict(X_test_data_replace)
print(y_pred)
example = pd.read_csv('../input/adult-pmr3508/sample_submission.csv')
example.head()

example['income']= y_pred
import csv
example.to_csv('predictions.csv',index=False)