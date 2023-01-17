import numpy as np

import sklearn as sk

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

#=====================

# Lê os dados

#=====================

data = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="NaN",index_col=0)

datatest = pd.read_csv("../input/adult-pmr3508/test_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="NaN", index_col=0)

data.describe()

data.head()
#==================================================

# Imprime a quantidade de dados faltantes nas colunas

#==================================================

for col_name, col_val in data.iteritems():

    missing_sum = sum(1 if val == "?" else 0 for val in col_val)

    print(col_name, missing_sum, "\n", data[col_name].unique())
#==================================================

# Substitui os dados faltantes por moda da coluna

#==================================================

for col in data:

    data[col] = data[col].fillna(data[col].mode())

for col in datatest:

    datatest[col] = datatest[col].fillna(datatest[col].mode())

data = data

datatest = datatest
#=================================================

# Encoda os dados STRING em classificação ordinal

#=================================================

stringfeatures = ["marital.status","education","occupation", "relationship", "race", "sex", "workclass", "native.country"]



numdata = data

numdata[stringfeatures] = numdata[stringfeatures].apply(preprocessing.LabelEncoder().fit_transform) 

numdatatest = datatest

numdatatest[stringfeatures] = numdatatest[stringfeatures].apply(preprocessing.LabelEncoder().fit_transform) 



#============================

# Normalização

#============================

normalizar = ["age", "education.num", "marital.status","occupation","relationship",

           "race", "sex", "native.country", "workclass"]



for col in normalizar:

    numdata[col] = (numdata[col] - numdata[col].min()) / (numdata[col].max() - numdata[col].min())

    numdatatest[col] = (numdatatest[col] - numdatatest[col].min()) / (numdatatest[col].max() - numdatatest[col].min())



numdata.head()
#==================================

# Seleção das Features Relevantes

#==================================

classes = ["age", "education.num", "marital.status","relationship","occupation",

           "sex", "capital.gain", "capital.loss"]

X = numdata.filter(items=classes)

Y = numdata.income
#=================================================

# Encontra o melhor K para KNN

#=================================================

maxk = 0

maxscore = 0

knn = None

maxknn = None

for k in range(26,27):

    knn = KNeighborsClassifier(k,n_jobs=-1)

    scores = cross_val_score(knn, X, Y, cv=10)

    print(f"{k} {scores.mean():.6f} {scores.max():.6f}")

    if scores.mean() > maxscore:

        maxscore = scores.mean()

        maxk = k

        maxknn = knn
#==========

# Fit

#==========

Xtest = numdatatest.filter(items=classes)

maxknn.fit(X,Y)
#=================================================

# Predict com a rede treinada

#=================================================

Ypred = maxknn.predict(Xtest)

Ypred = np.where(Ypred==0, "<=50K", Ypred) 

Ypred = np.where(Ypred=='1', ">50K", Ypred) 
#=================================================

# Gera o arquivo de saída

#=================================================

s = "Id,income\n"

for k,income in enumerate(Ypred):

    s += f"{k},{income}\n"



with open(f"submission.csv", "w+") as f:

    f.write(s)