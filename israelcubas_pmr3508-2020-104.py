import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



'''Centralização dos Gráficos Inline e Modificações de Plotagem'''

plt.rcParams.update({'font.family':'serif'})

plt.rcParams.update({'font.size':11})

from IPython.core.display import HTML as Center

Center(""" <style>.output_png { display: table-cell; text-align: center; vertical-align: middle; } </style> """)
df_train = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", index_col=['Id'], na_values='?')

df_train.head()
df_train.info()
df_train.describe()
df_train_an = df_train.copy()
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

df_train_an['income'] = LE.fit_transform(df_train_an['income'])

df_train_an['income']
plt.figure(figsize=(8,8))

sns.heatmap(df_train_an.corr(), square = True, annot=True, vmin=-1, vmax=1, cmap="RdGy")

plt.show()
plt.figure(figsize=(8,4))

sns.distplot(df_train_an['age'], color='maroon');

plt.title ("Idade", size='16')

plt.xlabel('Idade')

plt.ylabel('Densidade')

plt.grid(True)

plt.grid(which='major', linestyle='dashed', linewidth='0.5', color='silver')
plt.figure(figsize=(6,6))

sns.violinplot(x="income", y="hours.per.week", hue="sex", split=True, data=df_train_an);

plt.title ("Horas trabalhadas por Semana", size='16')

plt.xlabel(r'Renda - $0\;&\;1\;respectivamente\;<=50K\;&\;>50k$ ')

plt.ylabel('Horas')
plt.figure(figsize=(6,5))

sns.barplot(y="race", x="income", hue="sex", data=df_train_an);

plt.title (r"Renda Superior a $50k$ vs Raça vs Sexo", size='16')

plt.xlabel(r'Percentil de renda superior a $50k$')

plt.ylabel('Raça')
plt.figure(figsize=(6,10))

sns.barplot(y="native.country", x="income", data=df_train_an);

plt.title (r"Renda Superior a $50k$ vs País Nativo", size='16')

plt.xlabel(r'Percentil de renda superior a $50k$')

plt.ylabel('País Nativo')
plt.figure(figsize=(7,4))

df_train_an["native.country"].value_counts().plot(kind="bar", color="maroon")

plt.title ("Distribuição dos dados por País de Origem", size='16')

plt.xlabel('País')

plt.ylabel('Quantidade')
plt.figure(figsize=(6,5))

sns.barplot(x='income', y = 'marital.status', data = df_train_an)

plt.title (r"Renda Superior a $50k$ vs Estado Conjugal", size='16')

plt.xlabel(r'Percentil de renda superior a $50k$')

plt.ylabel('Estado Conjugal')
plt.figure(figsize=(6,5))

sns.barplot(x='income', y = 'relationship', data = df_train_an)

plt.title (r"Renda Superior a $50k$ vs Relação", size='16')

plt.xlabel(r'Percentil de renda superior a $50k$')

plt.ylabel('Relação')
df_train_an.drop_duplicates(keep='first', inplace=True)

df = df_train_an.drop(['fnlwgt', 'native.country', 'education'], axis=1)

df.head()
Y_train = df.pop('income')

X_train = df

X_train.head()
Val = X_train['workclass'].describe().top

X_train['workclass'] = X_train['workclass'].fillna(Val)

Val = X_train['occupation'].describe().top

X_train['occupation'] = X_train['occupation'].fillna(Val)

X_train
from sklearn.preprocessing import OneHotEncoder

from sklearn import preprocessing as prep



hot = prep.OneHotEncoder()

scaler = prep.StandardScaler()

robust = prep.RobustScaler()



num = list(X_train.select_dtypes(include=[np.number]).columns.values)

num.remove('capital.loss')

num.remove('capital.gain')

qua = list(X_train.select_dtypes(exclude = [np.number]).columns.values)

spr = ['capital.loss', 'capital.gain']

print("Atributos Numéricos: %s" %num)

print("Atributos Categóricos: %s" %qua)

print("Atributos Esparsos: %s" %spr)
from sklearn.compose import ColumnTransformer



preprocessor = ColumnTransformer(transformers = [('cat', hot, qua), ('num', scaler, num), ('spr', robust, spr)])

X_train = preprocessor.fit_transform(X_train)
import time

from sklearn.neighbors import KNeighborsClassifier as KNNC

from sklearn.model_selection import cross_val_score as CVS



def mean(score):

        soma = 0

        for i in range(len(score)):

            soma += 100*score[i]

        mean = soma/len(score)



        return mean



n = np.arange(22,33,1)

Scores = []



start = time.time()

for i in range(len(n)):

    knn = KNNC(n_neighbors = n[i], p = 2)

    score = CVS(knn, X_train, Y_train, cv = 5)

    Scores.append(mean(score))

end = time.time()

delta = end - start

minut = delta//60

seg = ((delta/60)-(delta//60))*60

    

best_k = 0

desv_pad = np.std(Scores)



for i in range(len(Scores)):

    if Scores[i] == max(Scores):

        best_k = n[i]

    

print("Melhor hiperparâmetro K encontrado: %d | Acurácia - Cross Validation: %.2f%% +/- %.2f%%" %(best_k, max(Scores), desv_pad))

print("Tempo de execução : %d minuto(s) e %d segundo(s)" %(minut, seg))



plt.figure(figsize=(7,5))

plt.scatter(n, Scores,c='darkslategray')

plt.plot(n, Scores,c='lightcoral')

plt.title ("Evolução da Acurácia em função de K", size='16')

plt.xlabel('Hiperparâmetro k')

plt.ylabel('Acurácia - Cross Validation')

plt.grid(True)

plt.minorticks_on()

plt.grid(which='major', linestyle='solid', linewidth='0.5', color='silver')

plt.grid(which='minor', linestyle='dashed', linewidth='0.5', color='lavender')
knn = KNNC(n_neighbors = 30, p = 2)

knn.fit(X_train, Y_train)
df_test = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values='?')

X_test = df_test.drop(['fnlwgt', 'native.country', 'education'], axis = 1)

X_test.info()
Val = X_test['occupation'].describe().top

X_test['occupation'] = X_test['occupation'].fillna(Val)

Val = X_test['workclass'].describe().top

X_test['workclass'] = X_test['workclass'].fillna(Val)

X_test = preprocessor.transform(X_test)
start = time.time()

Prdct = knn.predict(X_test)

end = time.time()

delta = end - start

minut = delta//60

seg = ((delta/60)-(delta//60))*60

Prdct = LE.inverse_transform(Prdct)

print("Tempo de predições da renda anual usando o KNN definido : %d minuto(s) e %d segundo(s)" %(minut, seg))

print(Prdct)
Sub = pd.DataFrame()

Sub[0] = df_test.index

Sub[1] = Prdct

Sub.columns = ['Id','income']

Sub.to_csv('submission.csv', index = False)

print(Sub)