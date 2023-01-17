import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn import preprocessing
adult_train = pd.read_csv("../input/adult-pmr3508/train_data.csv",index_col=['Id'],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

adult_test = pd.read_csv("../input/adult-pmr3508/test_data.csv",index_col=['Id'],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
adult_train.shape
adult_train.head()
Faltantes = adult_train.isnull().sum().sort_values(ascending = False)

Faltantes.head()
plt.figure(figsize=(17, 7))

adult_train.workclass.value_counts().plot(kind = 'pie')
plt.figure(figsize=(17, 7))

adult_train['native.country'].value_counts().plot(kind = 'pie')
plt.figure(figsize=(17, 7))

adult_train.occupation.value_counts().plot(kind = 'pie')
adult_train["workclass"] = adult_train["workclass"].fillna(adult_train["workclass"].describe().top)

adult_train['native.country'] = adult_train['native.country'].fillna(adult_train['native.country'].describe().top);

adult_train['occupation'] = adult_train['occupation'].fillna(adult_train['occupation'].describe().top);
Faltantes = adult_train.isnull().sum().sort_values(ascending = False)

Faltantes.head()
adult_test["workclass"] = adult_test["workclass"].fillna(adult_test["workclass"].describe().top)

adult_test['native.country'] = adult_test['native.country'].fillna(adult_test['native.country'].describe().top);

adult_test['occupation'] = adult_test['occupation'].fillna(adult_test['occupation'].describe().top);
Faltantes = adult_test.isnull().sum().sort_values(ascending = False)

Faltantes.head()
plt.figure(figsize=(15, 7))

adult_train.groupby("income").age.hist()

plt.legend(['<=50k','>50k'])

plt.xlabel('Age')

plt.ylabel('quantity')

plt.title('Age histogram')
plt.figure(figsize=(17, 7))

adult_train['education.num'].value_counts().plot(kind = 'pie')
plt.figure(figsize=(17, 7))

adult_train['marital.status'].value_counts().plot(kind = 'pie')
plt.figure(figsize=(17, 7))

adult_train['occupation'].value_counts().plot(kind = 'pie')
plt.figure(figsize=(17, 7))

adult_train['hours.per.week'].value_counts().plot(kind = 'pie')
plt.figure(figsize=(17, 7))

adult_train['capital.loss'].value_counts().plot(kind = 'pie')
plt.figure(figsize=(17, 7))

adult_train['capital.loss'].value_counts().plot(kind = 'pie')
plt.figure(figsize=(20, 7))

adult_train.groupby("income").workclass.hist()

plt.legend(['<=50k','>50k'])

plt.xlabel("Workclass")

plt.ylabel('quantity')

plt.title('Workclass histogram')
plt.figure(figsize=(20, 7))

adult_train.groupby("income").relationship.hist()

plt.legend(['<=50k','>50k'])

plt.xlabel("Relationship Status")

plt.ylabel('Relationship Status')

plt.title('Relationship histogram')
plt.figure(figsize=(20, 7))

adult_train.groupby("income").race.hist()

plt.legend(['<=50k','>50k'])

plt.xlabel("Race Status")

plt.ylabel('Race Status')

plt.title('Race histogram')
plt.figure(figsize=(20, 7))

adult_train.groupby("income").sex.value_counts().plot(kind = 'pie')

plt.legend(['<=50k','>50k'])

plt.xlabel("Sex Status")

plt.ylabel('Sex Status')

plt.title('Sex histogram')
adult_train_num=adult_train.apply(preprocessing.LabelEncoder().fit_transform)

adult_test_num=adult_test.apply(preprocessing.LabelEncoder().fit_transform)
adult_train_num.head()
rel = adult_train_num.drop(['income'],axis = 1)

rel_bi = adult_train_num['income']

rel = pd.concat([rel, rel_bi], axis = 1)

plt.figure(figsize=(15,8))

sns.heatmap(rel.corr(),annot=True)
trainX=adult_train[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]

trainY=adult_train.income

testX=adult_test[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
normalizador = preprocessing.StandardScaler()

trainX = normalizador.fit_transform(trainX)

testX = normalizador.fit_transform(testX)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

média=0

maior_média=0

melhor_k=0
for K in range(1,50):

    knn = KNeighborsClassifier(n_neighbors=K)

    scores = cross_val_score(knn,trainX,trainY,cv=10)

    média=scores.mean()

    if(média>maior_média):

        maior_média=média

        melhor_k=K
melhor_k

maior_média
knn = KNeighborsClassifier(n_neighbors=melhor_k)

knn.fit(trainX,trainY)
testY_res=knn.predict(testX)

testY_res
submiss=pd.DataFrame(testY_res,columns=['income'])

submiss.to_csv("PMR3508-2020-90_submissao.csv", index_label="Id")
from sklearn import preprocessing
trainX = adult_train_num[["age","education.num","capital.gain", "capital.loss", "hours.per.week","occupation",'relationship','race','sex','marital.status']]

trainY=adult_train.income

testX = adult_test_num[["age","education.num","capital.gain", "capital.loss", "hours.per.week","occupation",'relationship','race','sex','marital.status']]
normalizador = preprocessing.StandardScaler()

trainX = normalizador.fit_transform(trainX)

testX = normalizador.fit_transform(testX)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

média=0

maior_média=0

melhor_k=0

for K in range(1,50):

    knn = KNeighborsClassifier(n_neighbors=K)

    scores = cross_val_score(knn,trainX,trainY,cv=10)

    média=scores.mean()

    if(média>maior_média):

        maior_média=média

        melhor_k=K
melhor_k
knn = KNeighborsClassifier(n_neighbors=melhor_k)

knn.fit(trainX,trainY)
testY_res_num=knn.predict(testX)

testY_res_num
melhor_k
maior_média
submiss=pd.DataFrame(testY_res_num,columns=['income'])

submiss.to_csv("PMR3508-2020-90_submissao.csv", index_label="Id")