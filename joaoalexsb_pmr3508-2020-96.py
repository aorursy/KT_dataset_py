# Import das bibliotecas

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
# Import dos dados de treino e teste

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Lista das variáveis (para renomear as colunas)

var_labels = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

              "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

              "Hours per week", "Country", "Income"]



# Ler os dados de treino e teste, substituir o nome das variáveis e indentifcar os valores nulos como ?

df_train = pd.read_csv("../input/adult-pmr3508/train_data.csv",

                       names=var_labels,

                       header=0,

                       na_values="?")



# Remove o Income da lista de labels

var_labels.pop(-1)

df_test = pd.read_csv("../input/adult-pmr3508/test_data.csv",

                       index_col=0,

                       names=var_labels,

                       header=0,

                       na_values="?")
df_train.head()
df_train.shape
df_test.head()
df_test.shape
df_train.info()
df_train["Sex"].value_counts()
df_train["Education"].value_counts()
df_train.describe()
!pip install -U pandas-profiling
from pandas_profiling import ProfileReport

profile = ProfileReport(df_train, title='Exploratory Analysis',html={'style':{'full_width':True}})

profile
# Gráfico de pizza simples

plt.pie(df_train["Sex"].value_counts(), labels=df_train["Sex"].unique(),autopct='%1.0f%%')

plt.title('Sex');
# Gráfico de boxes da Idade quebrado renda (Income)

fig = sns.catplot(x="Income", y="Age", kind="boxen", data=df_train)

sns.set(rc={'figure.facecolor':'white'})

plt.title('Age vs. Income');
# Gráfico de boxes do tempo de Educação quebrado renda (Income)

fig = sns.catplot(x="Income", y="Education-Num", kind="boxen", data=df_train)

sns.set(rc={'figure.facecolor':'white'})

plt.title('Education vs. Income');
# Gráfico de boxes do Capital Gain quebrado pela renda (Income)

fig = sns.catplot(x="Income", y="Capital Gain", kind="boxen", data=df_train)

sns.set(rc={'figure.facecolor':'white'})

plt.title('Capital Gain vs. Income');
# Média do Capital Gain quebrada por Income

df_train.groupby(["Income"])["Capital Gain"].mean()
# Gráfico de boxes do Capital Loss quebrado pela renda (Income)

fig = sns.catplot(x="Income", y="Capital Loss", kind="boxen", data=df_train)

sns.set(rc={'figure.facecolor':'white'})

plt.title('Capital Loss vs. Income');
# Média do Capital Loss quebrada por Income

df_train.groupby(["Income"])["Capital Loss"].mean()
# Gráfico de boxes do Horas trabalhadas por semana quebrado pela renda (Income)

fig = sns.catplot(x="Income", y="Hours per week", kind="boxen", data=df_train)

sns.set(rc={'figure.facecolor':'white'})

plt.title('Hours per week vs. Income');
# Média do Capital Loss quebrada por Income

df_train.groupby(["Income"])["Hours per week"].mean()
# Gráfico de barras da Educação quebrado pela renda (Income)

fig, ax = plt.subplots(figsize=(10, 10))

fig = sns.countplot(y="Education", data=df_train, hue="Income", order = df_train["Education"].value_counts().index)

sns.set(rc={'figure.facecolor':'white'})

plt.title('Educação vs. Income');

# Gráfico de barras da Etnia (Race) quebrado pela renda (Income)

fig, ax = plt.subplots(figsize=(10, 10))

fig = sns.countplot(x="Race", data=df_train, hue="Income")

sns.set(rc={'figure.facecolor':'white'})

plt.title('Hours per week vs. Income');
df_train.shape
# Missing data

df_train.isnull().sum(axis = 0)
# Retirada da observação

df_train_na = df_train.dropna()

df_train_na.shape
# Treino

X_train = df_train[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]



y_train = df_train.Income



# Teste

X_test = df_test[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

# Import do KNN através do sklearn

from sklearn.neighbors import KNeighborsClassifier



# Instanciando o classificador

clf = KNeighborsClassifier(n_neighbors=30)
# Importando a validação cruzada

from sklearn.model_selection import cross_val_score



# Testando a acurrácia com a validação cruzada com 10 folds

score = cross_val_score(clf, X_train, y_train, cv=10)

score
# Acurrácia média

score.mean()
k = 30



acc = []

best_k = 0

best_acc = 0



for k in range(1,k+1):

    clf = KNeighborsClassifier(n_neighbors=k)

    score = cross_val_score(clf, X_train, y_train, cv=10)

    score_mean = round(score.mean(), 5)

    acc.append(score_mean)

    print("Number of k: ",k)

    print("Accuracy: ",score_mean)

    print("\n")

    

    if best_acc < score_mean:

        best_acc = score_mean

        best_k = k

        

print("Best k: ", best_k)

print("Best Accuracy", best_acc)
acc
plt.plot(range(1,k+1), acc)

plt.xlabel("k")

plt.ylabel("Accuracy")

plt.show()
# Criação do classificador e treino

clf = KNeighborsClassifier(n_neighbors=16)

clf.fit(X_train, y_train)



# Predição nos dados de teste

pred = clf.predict(X_test)

pred
submission = pd.DataFrame()
submission[0] = X_test.index

submission[1] = pred

submission.columns = ["Id", "Income"]

submission.head()
submission.to_csv('submission.csv',index = False)