import pandas as pd

import sklearn

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score as vdcruz

import seaborn as sns
train = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        engine='python',

        na_values="?")

test = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",na_values="?")
train.head()

index = test.Id
train["sex"].value_counts().plot(kind="bar")

train["education.num"].value_counts().plot(kind="bar")

train["relationship"].value_counts().plot(kind="bar")
train["education.num"].value_counts().plot(kind="box")
test.head()
for df in [train,test]:

    df.set_index('Id',inplace=True)
train.head()
total = train.isnull().sum().sort_values(ascending = False)

percent = ((train.isnull().sum()/train.isnull().count())*100).sort_values(ascending = False)

train_faltante = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

train_faltante.head()
Xtrain = train.drop(columns='income')

Ytrain = train.income

Ytrain.head()
Xtrain.head()
print(len(Xtrain))

print(len(Xtrain.dropna()))
for A in Xtrain.columns:

    Xtrain[A].fillna(Xtrain[A].mode()[0], inplace=True)

for A in test.columns:

    test[A].fillna(test[A].mode()[0], inplace=True)
Xtrain.shape

One_Hot_Xtrain = pd.get_dummies(Xtrain)

One_Hot_test = pd.get_dummies(test)

Xtrain, test = One_Hot_Xtrain.align(One_Hot_test,join='left',axis=1)

Xtrain.head()
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

Xtrain=sc_X.fit_transform(Xtrain)

test=sc_X.transform(test)
from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()

Xtrain = my_imputer.fit_transform(Xtrain)

test = my_imputer.transform(test)
Xtrain
knn = KNeighborsClassifier(n_neighbors=19)
resultado = vdcruz(knn, Xtrain, Ytrain, cv=10)

resultado.mean()
knn.fit(Xtrain,Ytrain)
Ytest = knn.predict(test)

Ytest
from sklearn.metrics import accuracy_score
accuracy_score(Ytrain,knn.predict(Xtrain))
subm11 =pd.DataFrame()
subm11['Id']=index

subm11['income']=Ytest
subm11.head()
subm11.to_csv("subm1copy.csv",index=False)
train = pd.read_csv("/kaggle/input/costarica/train.csv", engine="python")

test = pd.read_csv("/kaggle/input/costarica/test.csv",engine="python")
index = test.Id

train.head()
test.head()
for df in [train,test]:

    df.set_index('Id',inplace=True)
total = train.isnull().sum().sort_values(ascending = False)

percent = ((train.isnull().sum()/train.isnull().count())*100).sort_values(ascending = False)

train_faltante = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

train_faltante.head()
train.drop(columns=["rez_esc","v18q1","v2a1"]).head()
Xtrain = train.drop(columns='Target')

Ytrain = train.Target

Ytrain.head()
Xtrain = train.drop(columns='Target')

Ytrain = train.Target

Ytrain.head()
Xtrain.head()
print(len(Xtrain))

print(len(Xtrain.dropna())) #nao da pra tirar os dados faltantes, vou substituir pela moda
for A in Xtrain.columns:

    Xtrain[A].fillna(Xtrain[A].mode()[0], inplace=True)

for A in test.columns:

    test[A].fillna(test[A].mode()[0], inplace=True)
One_Hot_Xtrain = pd.get_dummies(Xtrain)

One_Hot_test = pd.get_dummies(test)

Xtrain, test = One_Hot_Xtrain.align(One_Hot_test,join='left',axis=1)

Xtrain.head()
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

Xtrain=sc_X.fit_transform(Xtrain)

test=sc_X.transform(test)

from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()

Xtrain = my_imputer.fit_transform(Xtrain)

test = my_imputer.transform(test)
Xtrain
knn = KNeighborsClassifier(n_neighbors=50)
resultado = vdcruz(knn, Xtrain, Ytrain, cv=10)

resultado.mean()
knn.fit(Xtrain,Ytrain)
Ytest = knn.predict(test)

Ytest
from sklearn.metrics import accuracy_score
accuracy_score(Ytrain,knn.predict(Xtrain))

subm11 =pd.DataFrame()
subm11['Id']=index

subm11['target']=Ytest
subm11.head()
subm11.to_csv("submcostarica.csv",index=False)