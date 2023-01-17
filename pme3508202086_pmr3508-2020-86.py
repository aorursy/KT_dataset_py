import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import sklearn



%matplotlib inline
#test = pd.read_csv("test_data.csv", index_col = 'Id', na_values="?")

test = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values="?")

#train = pd.read_csv("train_data.csv",  index_col = 'Id', na_values="?")

train = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", index_col=['Id'], na_values="?")

train.columns.values
train.head()
train.shape
train.info()
test.head()
test.shape
test.info()
total = train.isnull().sum().sort_values(ascending = False)

percent = ((train.isnull().sum()/train.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 'columns', keys = ['Total', '%'])

missing_data.head()
train.describe()
plt.figure(figsize=(12,6))

train['capital.gain'].hist()

plt.xlabel('capital gain')

plt.ylabel('quantity')

plt.title('Capital gain histogram')
plt.figure(figsize=(12,6))

train['capital.loss'].hist(color = 'coral')

plt.xlabel('capital loss')

plt.ylabel('quantity')

plt.title('Capital loss histogram')
plt.figure(figsize=(12,6))

sns.boxenplot(x='income', y='age', data=train, palette='viridis')

plt.title('Age boxenplot')
plt.figure(figsize=(12,6))

train['age'].hist(color = 'mediumseagreen')

plt.xlabel('age')

plt.ylabel('quantity')

plt.title('Age histogram')

plt.axvline(train['age'].mean(), color='k', linestyle='dashed', linewidth=1)
plt.figure(figsize=(10,8))

train['hours.per.week'].hist(color = 'mediumseagreen', bins = 20)

plt.xlabel('hours.per.week')

plt.ylabel('quantity')

plt.title('Hours.per.week histogram')

plt.axvline(train['hours.per.week'].mean(), color='k', linestyle='dashed', linewidth=1)
trabalhamuito = train[train['hours.per.week'] > 85]['income'] == "<=50K"



round((trabalhamuito).sum()/trabalhamuito.count(),2)
plt.figure(figsize=(12,6))

train['education.num'].hist(color = 'mediumseagreen')

plt.xlabel('education.num')

plt.ylabel('quantity')

plt.title('Education.num histogram')
from sklearn.preprocessing import LabelEncoder

train_encoded = train.copy()

le = LabelEncoder()

train_encoded['income'] = le.fit_transform(train_encoded['income'])
sns.catplot(y='workclass', x='income', kind='bar', data = train_encoded, height=8)

plt.title('Workclass catplot')
sns.catplot(y='education', x='income', kind='bar', data=train_encoded, palette='viridis', height=8)

plt.title('Education catplot')
sns.catplot(y='marital.status', x='income', kind='bar', data=train_encoded, height=8)

plt.title('Marital.status catplot')
sns.catplot(y='occupation', x='income', kind='bar', data=train_encoded, height=8)

plt.title('Occupation catplot')
sns.catplot(y='relationship', x='income', kind='bar', data=train_encoded, height=8)

plt.title('Relationship catplot')
sns.catplot(y='race', x='income', kind='bar', data=train_encoded, height=8)

plt.title('Race catplot')
plt.figure(figsize=(10, 8))

train['race'].value_counts().plot(kind = 'pie')
pd.concat([train[train['income'] == '<=50K']['race'].value_counts(), train[train['income'] == '>50K']['race'].value_counts()], axis=1, keys=["<=50K", ">50K"]).plot(kind='bar', figsize=(10,8))

plt.xlabel('race')

plt.ylabel('quantity')

plt.title('Race and income comparation')
plt.figure(figsize=(10,8))

sns.catplot(y='native.country', x='income', kind='bar', data=train_encoded, palette='viridis', height=8);

plt.title('Native.country catplot')


plt.figure(figsize=(10,8))

train['native.country'].value_counts().plot(kind = 'bar')
sns.catplot(y='sex', x='income', kind='bar', data=train_encoded, height=8)

plt.title('Sex catplot')
plt.figure(figsize=(12, 6))

train['sex'].value_counts().plot(kind = 'pie')
pd.concat([train[train['income'] == '<=50K']['sex'].value_counts(), train[train['income'] == '>50K']['sex'].value_counts()], axis=1, keys=["<=50K", ">50K"]).plot(kind='bar', figsize=(12,6))

plt.xlabel('sex')

plt.ylabel('quantity')

plt.title('Sex and income comparation')
train_data = train.drop(columns=['fnlwgt', 'native.country'])

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
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

vizinhos = [18,19,20,21,22,23]

maior = 0

parametro = 0

for k in vizinhos:

    score = cross_val_score(KNeighborsClassifier(n_neighbors=k), train_data, var_classe, cv = 5, scoring="accuracy").mean()

    if score > maior:

        maior = score

        parametro =k

print("Melhor parâmetro:",parametro)

print("Maior pontuação:",maior)


test = pd.read_csv("test_data.csv", index_col = ['Id'], na_values="?")

knn = KNeighborsClassifier(n_neighbors=20)

knn.fit(train_data,var_classe)
teste = test.drop(columns=['fnlwgt', 'native.country'])
teste = preprocessador.transform(teste)

predicao = knn.predict(teste)
entrega = pd.DataFrame()

entrega[0] = test_data.index

entrega[1] = predicao

entrega.columns = ['Id','income']

entrega.to_csv('submission.csv',index = False)