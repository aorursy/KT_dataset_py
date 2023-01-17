import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn
treino = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?",

        index_col=['Id'])
treino.head()
treino.info()
treino.describe()
treino_copia = treino.copy()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

treino_copia['income'] = le.fit_transform(treino_copia['income'])
sns.pairplot(treino, diag_kws={'bw':"1.0"}, hue = 'income', palette='viridis')
plt.figure(figsize=(10,8))

sns.heatmap(treino_copia.corr(), vmin=-1, vmax=1, annot=True, cmap='viridis')

plt.show()
plt.figure(figsize=(10,8))

sns.boxenplot(x='income', y='education.num', data=treino, palette='viridis')

plt.title('Education.num boxenplot')
plt.figure(figsize=(10,8))

treino['education.num'].hist(color = 'mediumseagreen')

plt.xlabel('education.num')

plt.ylabel('quantity')

plt.title('Education.num histogram')
plt.figure(figsize=(10,8))

sns.boxenplot(x='income', y='age', data=treino, palette='viridis')

plt.title('Age boxenplot')
plt.figure(figsize=(10,8))

treino['age'].hist(color = 'mediumseagreen')

plt.xlabel('age')

plt.ylabel('quantity')

plt.title('Age histogram')
plt.figure(figsize=(10,8))

sns.boxenplot(x='income', y='hours.per.week', data=treino, palette='viridis')

plt.title('Hours.per.week boxenplot')
plt.figure(figsize=(10,8))

treino['hours.per.week'].hist(color = 'mediumseagreen')

plt.xlabel('hours.per.week')

plt.ylabel('quantity')

plt.title('Hours.per.week histogram')
plt.figure(figsize=(10,8))

treino['capital.gain'].hist(color = 'mediumseagreen')

plt.xlabel('capital.gain')

plt.ylabel('quantity')

plt.title('Capital.gain histogram')
plt.figure(figsize=(10,8))

treino['capital.loss'].hist(color = 'mediumseagreen')

plt.xlabel('capital.loss')

plt.ylabel('quantity')

plt.title('Capital.loss histogram')
sns.catplot(y='workclass', x='income', kind='bar', data=treino_copia, palette='viridis', height=8)

plt.title('Workclass catplot')
sns.catplot(y='education', x='income', kind='bar', data=treino_copia, palette='viridis', height=8)

plt.title('Education catplot')
sns.catplot(y='marital.status', x='income', kind='bar', data=treino_copia, palette='viridis', height=8)

plt.title('Marital.status catplot')
sns.catplot(y='occupation', x='income', kind='bar', data=treino_copia, palette='viridis', height=8)

plt.title('Occupation catplot')
sns.catplot(y='relationship', x='income', kind='bar', data=treino_copia, palette='viridis', height=8)

plt.title('Relationship catplot')
sns.catplot(y='race', x='income', kind='bar', data=treino_copia, palette='viridis', height=8)

plt.title('Race catplot')
plt.figure(figsize=(10, 8))

colors = ['darkslateblue', 'teal', 'darkcyan', 'mediumseagreen', 'yellowgreen', ]

treino['race'].value_counts().plot(kind = 'pie', colors=colors)
income_menor = treino.loc[treino['income'] == '<=50K']

income_maior = treino.loc[treino['income'] == '>50K']



pd.concat([income_menor['race'].value_counts(), income_maior['race'].value_counts()], axis=1, keys=["<=50K", ">50K"]).plot(kind='bar', color=colors, figsize=(10,8))

plt.xlabel('race')

plt.ylabel('quantity')

plt.title('Race and income comparation')
sns.catplot(y='sex', x='income', kind='bar', data=treino_copia, palette='viridis', height=8)

plt.title('Sex catplot')
plt.figure(figsize=(10, 8))

treino['sex'].value_counts().plot(kind = 'pie', colors=colors)
pd.concat([income_menor['sex'].value_counts(), income_maior['sex'].value_counts()], axis=1, keys=["<=50K", ">50K"]).plot(kind='bar', color=colors, figsize=(10,8))

plt.xlabel('sex')

plt.ylabel('quantity')

plt.title('Sex and income comparation')
plt.figure(figsize=(10,8))

sns.catplot(y='native.country', x='income', kind='bar', data=treino_copia, palette='viridis', height=8);

plt.title('Native.country catplot')
treino["native.country"].value_counts()
treino.drop_duplicates(keep='first', inplace=True)
treino = treino.drop(['fnlwgt', 'native.country', 'education'], axis=1)
Y_treino = treino.pop('income')

X_treino = treino
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder



pipeline_categ = Pipeline(steps = [

    ('imputador', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(sparse=False))

])
from sklearn.preprocessing import StandardScaler

from sklearn.impute import KNNImputer



pipeline_num = Pipeline(steps = [

    ('imputer', KNNImputer(n_neighbors=5, weights="uniform")),

    ('scaler', StandardScaler())

])
from sklearn.preprocessing import RobustScaler



pipeline_robust = Pipeline(steps = [

    ('imputer', KNNImputer(n_neighbors=5, weights="uniform")),

    ('scaler', RobustScaler())

])
from sklearn.compose import ColumnTransformer



colunasNumericas = list(X_treino.select_dtypes(include = [np.number]).columns.values)

colunasCategoricas = list(X_treino.select_dtypes(exclude = [np.number]).columns.values)



colunasNumericas.remove('capital.gain')

colunasNumericas.remove('capital.loss')



preprocessador = ColumnTransformer(transformers = [

    ('numerico', pipeline_num, colunasNumericas),

    ('categorico', pipeline_categ, colunasCategoricas),

    ('robust', pipeline_robust, ['capital.gain', 'capital.loss'])

])
X_treino = preprocessador.fit_transform(X_treino)
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier



melhorK = 10

melhorAcuracia = 0.0



print('- Busca grosseira')

for k in [10, 20, 30]:

    acuracia = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_treino, Y_treino, cv=5, scoring='accuracy').mean()

    print('k =', k,': Acurácia',100 * acuracia)

    if acuracia > melhorAcuracia:

        melhorAcuracia = acuracia

        melhorK = k

print('========')



print('- Busca fina')

for k in range((melhorK - 5), (melhorK + 5)):

    acuracia = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_treino, Y_treino, cv=5, scoring='accuracy').mean()

    print('k =', k,': Acurácia',100 * acuracia)

    if acuracia > melhorAcuracia:

        melhorAcuracia = acuracia

        melhorK = k

print('========')



print('Melhor k:', melhorK)

print('Melhor acurácia:', 100 * melhorAcuracia)
knn = KNeighborsClassifier(n_neighbors=26)

knn.fit(X_treino, Y_treino)
teste = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")

X_teste = teste.drop(['fnlwgt', 'native.country', 'education'], axis=1)

X_teste = preprocessador.transform(X_teste)

predicoes = knn.predict(X_teste)



predicoes
submissao = pd.DataFrame()

submissao[0] = teste.index

submissao[1] = predicoes

submissao.columns = ['Id', 'income']



submissao.to_csv('submission.csv', index = False)