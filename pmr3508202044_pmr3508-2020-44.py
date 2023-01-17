import pandas as pd                     # biblioteca para manipulação e análise de dados

import sklearn                          # biblioteca para aprendizado de máquina

import seaborn as sns                   # biblioteca para visualização de dados

import numpy as np                      # biblioteca para manipulação de arrays e matrizes

import matplotlib.pyplot as plt         # biblioteca para visualização de dados

df_train = pd.read_csv("../input/adult-pmr3508/train_data.csv", index_col=['Id'], na_values="?")

df_train.head()
df_train.info()
df_train.describe()
print('occupation: \n')

print(df_train['occupation'].describe(), '\n \n')

print('workclass: \n')

print(df_train['workclass'].describe(), '\n \n')

print('native country: \n')

print(df_train['native.country'].describe(), '\n \n')
# o valor .describe().top é valor da moda em cada coluna, dada pela função .describe)()



moda_occupation = df_train['occupation'].describe().top

moda_workclass = df_train['workclass'].describe().top

moda_native_country = df_train['native.country'].describe().top



# a função .fillna() substitui os dados faltantes pelo valor fornecido



df_train['occupation'] = df_train['occupation'].fillna(moda_occupation)

df_train['workclass'] = df_train['workclass'].fillna(moda_workclass)

df_train['native.country'] = df_train['native.country'].fillna(moda_native_country)
df_train.info()
# a função .replace() substitui as strings de renda pelos números 0 ou 1



df_train['income'] = df_train['income'].replace('<=50K',0)

df_train['income'] = df_train['income'].replace('>50K',1)
plt.figure(figsize=(10,10))



# a função triu() formata a matriz de correlação, deixando-a sem a diagonal principal e os valores repetidos acima dela

triangular = np.triu(np.ones_like(df_train.corr(), dtype=np.bool))





sns.heatmap(df_train.corr(), mask = triangular, square=True, annot=True, vmin=-1, vmax=1, cmap="seismic")

plt.show()
# o método .drop() retira uma linha ou coluna do dataframe (no nosso caso, uma coluna)

df_train = df_train.drop('fnlwgt', axis = 1)
sns.catplot(y="workclass", x="income", kind="bar", data = df_train)
edu = df_train[["education", "education.num"]]

edu.drop_duplicates()
sns.catplot(y="education", x="income", kind="bar", data = df_train)
sns.catplot(y="marital.status", x="income", kind="bar", data = df_train)
sns.catplot(y="occupation", x="income", kind="bar", data = df_train)
sns.catplot(y="relationship", x="income", kind="bar", data = df_train)
sns.catplot(y="race", x="income", kind="bar", data = df_train)
sns.catplot(y="sex", x="income", kind="bar", data = df_train)
sns.catplot(y="native.country", x="income", kind="bar", data = df_train)
df_train["native.country"].value_counts()
df_train = df_train.drop(["native.country", "education"], axis=1)
rotulo_train = df_train.pop('income')

atributo_train = df_train

# cria lista com as colunas numéricas

num_col = list(atributo_train.select_dtypes(include=[np.number]).columns.values)



# cria lista com as colunas categóricas

cat_col = list(atributo_train.select_dtypes(exclude=[np.number]).columns.values)

from sklearn.pipeline import  Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



pipeline_cat = Pipeline(steps = [

    ('onehot', OneHotEncoder(drop="if_binary"))

])
from sklearn.preprocessing import StandardScaler



pipeline_num = Pipeline(steps = [

    ('scaler', StandardScaler())

])
from sklearn.compose import ColumnTransformer



preprocessador = ColumnTransformer(transformers = [

    ('num', pipeline_num, num_col),

    ('cat', pipeline_cat, cat_col)

])



atributo_train = preprocessador.fit_transform(atributo_train)
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier



k_escolhido = 0

maior_pontuacao = 0

teste_k = range(15,31)



for k in teste_k:    # testa os k's de 15 a 30

    pontuacao = cross_val_score(KNeighborsClassifier(n_neighbors=k), atributo_train, rotulo_train, cv=5, scoring="accuracy").mean()    #calcula a acuracia de cada k

    if pontuacao > maior_pontuacao:     # compara com a maior acuracia até o momento e, se for maior, esse k vira o novo k_escolhido

        maior_pontuacao = pontuacao

        k_escolhido = k

        

print('Melhor k: {}'.format(k_escolhido))

print('Acuracia: {}'.format(maior_pontuacao))
df_test = pd.read_csv("../input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values="?")

df_test = df_test.drop(["fnlwgt", "native.country", "education"], axis=1)

# o valor .describe().top é valor da moda em cada coluna, dada pela função .describe)()



moda_occupation = df_test['occupation'].describe().top

moda_workclass = df_test['workclass'].describe().top



# a função .fillna() substitui os dados faltantes pelo valor fornecido



df_test['occupation'] = df_test['occupation'].fillna(moda_occupation)

df_test['workclass'] = df_test['workclass'].fillna(moda_workclass)



atributo_test = df_test

atributo_test = preprocessador.transform(atributo_test)
kNN = KNeighborsClassifier(n_neighbors=21)

kNN.fit(atributo_train, rotulo_train)



predicoes = kNN.predict(atributo_test)



print(predicoes)
submissao = pd.DataFrame()

submissao[0] = df_test.index

submissao[1] = predicoes

submissao.columns = ['Id', 'income']



# a função .replace() substitui as de volta as strings



submissao['income'] = submissao['income'].replace(0, '<=50K')

submissao['income'] = submissao['income'].replace(1, '>50K')



submissao.to_csv('submission.csv',index = False)