import numpy as np

import pandas as pd

train_data = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", index_col=['Id'], na_values="?")

train_data.head()
train_analise = train_data.copy()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_analise['income'] = le.fit_transform(train_analise['income'])

import matplotlib.pyplot as plt

import seaborn as sns



variavel_num = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]

for n in variavel_num:

    sns.catplot(x="income", y=n, kind="boxen", data=train_analise)
train_analise.describe()
variavel_cat = ["workclass", "education", "marital.status", "occupation", "relationship", "race", "sex", "native.country"]

for n in variavel_cat:

    sns.catplot(y=n, x="income", kind="bar", data=train_analise)
train_data = train_data.drop(columns=['fnlwgt', 'native.country'])

var_classe = train_data.pop('income')
var_num = list(train_data.select_dtypes(include=[np.number]).columns.values)

var_num.remove('capital.gain')

var_num.remove('capital.loss')



var_esp = ['capital.gain', 'capital.loss']



var_cat = list(train_data.select_dtypes(exclude=[np.number]).columns.values)
from sklearn.pipeline import Pipeline

from sklearn.impute import KNNImputer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer





num_pipe = Pipeline(steps = [

    ('imputer', KNNImputer(n_neighbors=15, weights="uniform")),

    ('scaler', StandardScaler())

])



esp_pipe = Pipeline(steps = [

    ('imputer', KNNImputer(n_neighbors=15, weights="uniform")),

    ('scaler', RobustScaler())

])



cat_pipe = Pipeline(steps = [

    ('imputer', SimpleImputer(strategy = 'most_frequent')),

    ('onehot', OneHotEncoder(drop='if_binary'))

])



preprocessador = ColumnTransformer(transformers = [

    ('num', num_pipe, var_num),

    ('spr', esp_pipe, var_esp),

    ('cat', cat_pipe, var_cat)

])



train_data = preprocessador.fit_transform(train_data)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

vizinhos = [10,15,20,25,30]

maior = 0

parametro = 0

for k in vizinhos:

    score = cross_val_score(KNeighborsClassifier(n_neighbors=k), train_data, var_classe, cv = 5, scoring="accuracy").mean()

    if score > maior:

        maior = score

        parametro =k

print("Melhor parâmetro:",parametro)

print("Maior pontuação:",maior)

        
knn = KNeighborsClassifier(n_neighbors=20)

knn.fit(train_data,var_classe)



test_data = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values="?")

teste = test_data.drop(columns=['fnlwgt', 'native.country'])

teste = preprocessador.transform(teste)



predicao = knn.predict(teste)
submission = pd.DataFrame()

submission[0] = test_data.index

submission[1] = predicao

submission.columns = ['Id','income']

submission.to_csv('submission.csv',index = False)