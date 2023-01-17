import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
adult = pd.read_csv("../input/adult-pmr3508/train_data.csv",index_col=['Id'], na_values="?")

test_adult = pd.read_csv("../input/adult-pmr3508/test_data.csv",index_col=['Id'], na_values="?")
adult.shape
adult.head(10)
adult.info()
adult.describe()
adult.describe(include='object')
duplicados = adult[adult.duplicated(keep='first')]

print(duplicados)
adult.drop_duplicates(keep='first', inplace=True) 
adult_analysis = adult.copy()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

adult_analysis['income'] = le.fit_transform(adult['income'])
adult_analysis['income']
mask = np.triu(np.ones_like(adult_analysis.corr(), dtype=np.bool))



plt.figure(figsize=(8,8))



sns.heatmap(adult_analysis.corr(), mask=mask, square = True, annot=True, vmin=-1, vmax=1, cmap='spring')

plt.show()
sns.distplot(adult['age'])
sns.catplot(x="income", y="age", kind="boxen", data=adult_analysis)
sns.catplot(x="income", y="education.num", kind="boxen", data=adult_analysis)
sns.catplot(x="income", y="hours.per.week", kind="boxen", data=adult_analysis)
sns.catplot(x="income", y="capital.gain", kind="boxen", data=adult_analysis)
sns.catplot(x="income", y="capital.loss", kind="boxen", data=adult_analysis)
plt.figure(figsize=(10, 7))

adult['capital.gain'].hist(color = 'coral')

plt.xlabel('capital gain')

plt.ylabel('quantity')

plt.title('Capital gain histogram')
plt.figure(figsize=(10, 7))

adult['capital.loss'].hist(color = 'coral')

plt.xlabel('capital loss')

plt.ylabel('quantity')

plt.title('Capital loss histogram')
adult_analysis.describe()
adult_analysis["education"].value_counts().plot(kind="bar")
sns.catplot(y="education", x="income", kind="bar", data=adult_analysis)
adult_analysis["workclass"].value_counts().plot(kind="bar")
sns.catplot(y="workclass", x="income", kind="bar", data=adult_analysis)
adult["occupation"].value_counts().plot(kind="bar")
sns.catplot(y="occupation", x="income", kind="bar", data=adult_analysis)
adult["relationship"].value_counts().plot(kind="bar")
sns.catplot(y="relationship", x="income", kind="bar", data=adult_analysis)
adult["marital.status"].value_counts().plot(kind="bar")
sns.catplot(y="marital.status", x="income", kind="bar", data=adult_analysis)
adult["race"].value_counts().plot(kind="pie")
sns.catplot(y="race", x="income", kind="bar", data=adult_analysis)
adult["sex"].value_counts().plot(kind="pie")
sns.catplot(y="sex", x="income", kind="bar", data=adult_analysis)
adult["native.country"].value_counts().plot(kind="bar")
adult = adult.drop(['fnlwgt', 'native.country', 'education'], axis=1)
test_adult = test_adult.drop(['fnlwgt', 'native.country', 'education'], axis=1)
adult.head()
adult.shape
adult.info()
colunas_com_dados_faltantes = ['workclass', 'occupation']

adult.replace('?', np.nan, inplace=True)

for col in colunas_com_dados_faltantes:

    adult[col] = adult[col].fillna(adult[col].mode()[0])

    assert adult[col].isnull().any() == False
test_adult.replace('?', np.nan, inplace=True)

for col in colunas_com_dados_faltantes:

    test_adult[col] = test_adult[col].fillna(test_adult[col].mode()[0])

    assert test_adult[col].isnull().any() == False
Y = adult.pop("income")

X = adult
X_teste = test_adult
features_num = list(X.select_dtypes(include=[np.number]).columns.values)

features_num
features_cat =list(X.select_dtypes(exclude=[np.number]).columns.values)

features_cat
dummies = pd.get_dummies(X[features_cat])

X_num = X[features_num]

X = pd.merge(X_num, dummies, left_index=True, right_index=True)
X.head()
dummies = pd.get_dummies(X_teste[features_cat])

X_testenum = X_teste[features_num]

X_teste = pd.merge(X_testenum, dummies, left_index=True, right_index=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)
X_teste = scaler.fit_transform(X_teste)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=20)
from sklearn.model_selection import cross_val_score

score = cross_val_score(knn, X, Y, cv = 5, scoring="accuracy")

print("Acurácia com cross validation:", score.mean())
n_vizinhos = [15,18,22,26,30]

scores = {}

for n in n_vizinhos:

  score = cross_val_score(KNeighborsClassifier(n_neighbors=n), X, Y, cv = 5, scoring="accuracy").mean()

  scores[n] = score

melhor_n = max(scores, key=scores.get)

print("Melhor n: ", melhor_n)

print("Melhor acurácia: ", scores[melhor_n])
knn = KNeighborsClassifier(n_neighbors=26)

knn.fit(X,Y)
Y_teste_predict = knn.predict(X_teste)
sub = pd.DataFrame()

sub[0] = test_adult.index

sub[1] = Y_teste_predict

sub.columns = ['Id','income']

sub.to_csv('PMR-3508-2020-169.csv',index = False)
sub.head()