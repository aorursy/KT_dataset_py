import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

import seaborn as sns



from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
adult = pd.read_csv("../input/adult-pmr3508/train_data.csv",

          header=0,

          sep=r'\s*,\s*',

          engine='python',

          na_values="?")
adult.head()
adult.describe(include='all')
adult_test = pd.read_csv("../input/adult-pmr3508/test_data.csv",

          header=0,

          sep=r'\s*,\s*',

          engine='python',

          na_values="?")
adult_test.head()
adult_test.describe(include='all')
adult['age'].hist()

plt.xlabel("Idade")

plt.ylabel("Frequência")

plt.title("Histograma da idade")
adult[adult['income'] == '<=50K']['age'].hist(label="<=50K")

adult[adult['income'] == '>50K']['age'].hist(label=">50K")

plt.xlabel("Idade")

plt.ylabel("Frequência")

plt.title("Discriminação da idade por faixa de ganhos")

plt.legend()



print("Média de idade (<=50K):", adult[adult['income'] == '<=50K']['age'].mean())

print("Média de idade (>50K):", adult[adult['income'] == '>50K']['age'].mean())
adult['education'].value_counts().plot(kind='bar')

plt.xlabel("Nível de educação")

plt.ylabel("Frequência")

plt.title("Nível educacional no dataset")
adult['native.country'].value_counts().plot(kind='pie')

plt.xlabel("País de origem")

plt.ylabel("Frequência")

plt.title("País de origem do dataset")
s = adult['native.country'].copy()

s[s != "United-States"] = "Other"
s.value_counts().plot(kind='pie', autopct='%1.0f%%')

plt.title("País de origem agrupando os imigrantes")
s[adult['income'] == '<=50K'].value_counts().plot(kind='pie', autopct='%1.0f%%')

plt.title("País de origem (<=50K)")
s[adult['income'] == '>50K'].value_counts().plot(kind='pie', autopct='%1.0f%%')

plt.title("País de origem (>50K)")
adult['sex'].value_counts().plot(kind='pie', autopct='%1.0f%%')

plt.title("Distribuição de sexo no dataset")
adult[adult['income'] == '<=50K']['sex'].value_counts().plot(kind='pie', autopct='%1.0f%%')

plt.title("Distribuição de sexo (<=50K)")
adult[adult['income'] == '>50K']['sex'].value_counts().plot(kind='pie', autopct='%1.0f%%')

plt.title("Distribuição de sexo (>50K)")
adult['workclass'].value_counts().plot(kind='pie', autopct='%1.0f%%')

plt.title("Distribuição da workclass")
adult['occupation'].value_counts().plot(kind='pie', autopct='%1.0f%%')

plt.title("Distribuição das profissões")

print("Número de profissões:", len(adult['occupation'].value_counts()))
adult.isna().sum()
adult_test.isna().sum()
adult['native.country'] = adult['native.country'].fillna(adult.mode().iloc[0].loc['native.country'])

adult_test['native.country'] = adult_test['native.country'].fillna(adult.mode().iloc[0].loc['native.country'])



adult['workclass'] = adult['workclass'].fillna(adult.mode().iloc[0].loc['workclass'])

adult_test['workclass'] = adult_test['workclass'].fillna(adult.mode().iloc[0].loc['workclass'])
adult_com_occupation = adult.copy()

adult_test_com_occupation = adult_test.copy()
adult = adult.drop("occupation", axis=1)

adult_test = adult_test.drop("occupation", axis=1)
adult.isna().sum()
adult_test.isna().sum()
corr = adult.apply(preprocessing.LabelEncoder().fit_transform).corr()

f, ax = plt.subplots(figsize=(20, 13))

sns.heatmap(corr, vmax=.7, square=True, cmap="coolwarm", annot = True)
def evaluate_model(X_train, Y_train, k):

  knn = KNeighborsClassifier(n_neighbors=k)

  score = cross_val_score(knn, X_train, Y_train, cv=10)

  mean_acc = score.mean()

  max_acc = score.max()

  min_acc = score.min()

  print("KNN com validação cruzada em 10 folds\n"+

        f"\tAcurácia média: {mean_acc}\n"+

        f"\tAcurácia máxima: {max_acc}\n"+

        f"\tAcurácia mínima: {min_acc}")

X_train = adult.drop(['Id', 'income'], axis=1).copy()
X_train = X_train.apply(preprocessing.LabelEncoder().fit_transform)
X_train.head()
Y_train = adult['income']
Y_train.head()
evaluate_model(X_train, Y_train, k=3)
adult_com_occupation = adult_com_occupation.dropna()



X_train = adult_com_occupation.drop(['Id', 'income'], axis=1).copy()

Y_train = adult_com_occupation['income']
X_train = X_train.apply(preprocessing.LabelEncoder().fit_transform)

evaluate_model(X_train, Y_train, 3)
Y_train = adult['income']
X_train = adult.drop(['Id', 'income', 'fnlwgt'], axis=1).copy()

X_train = X_train.apply(preprocessing.LabelEncoder().fit_transform)

X_train.head()
evaluate_model(X_train, Y_train, 3)
X_train = adult.drop(['Id', 'income', 'fnlwgt', 'workclass'], axis=1).copy()

X_train = X_train.apply(preprocessing.LabelEncoder().fit_transform)

X_train.head()
evaluate_model(X_train, Y_train, 3)
print("Removendo 'education' e 'workclass'")

X_train = adult.drop(['Id', 'income', 'fnlwgt', 'workclass', 'education'], axis=1).copy()

X_train = X_train.apply(preprocessing.LabelEncoder().fit_transform)

evaluate_model(X_train, Y_train, 3)
print("Removendo 'education.num' e 'workclass")

X_train = adult.drop(['Id', 'income', 'fnlwgt', 'workclass', 'education.num'], axis=1).copy()

X_train = X_train.apply(preprocessing.LabelEncoder().fit_transform)

evaluate_model(X_train, Y_train, 3)
print("Removendo 'education', mas mantendo 'workclass'")

X_train = adult.drop(['Id', 'income', 'fnlwgt', 'education'], axis=1).copy()

X_train = X_train.apply(preprocessing.LabelEncoder().fit_transform)

evaluate_model(X_train, Y_train, 3)
print("Removendo 'native.country'")

X_train = adult.drop(['Id', 'income', 'fnlwgt', 'education', 'native.country'], axis=1).copy()

X_train = X_train.apply(preprocessing.LabelEncoder().fit_transform)

evaluate_model(X_train, Y_train, 3)
print("Discriminando native.country entre United-States e Other")

X_train = adult.drop(['Id', 'income', 'fnlwgt', 'education'], axis=1).copy()

X_train['native.country'][X_train['native.country'] != "United-States"] = "Other"

X_train = X_train.apply(preprocessing.LabelEncoder().fit_transform)

evaluate_model(X_train, Y_train, 3)
print("Utilizando o StandardScaler nos dados de entrada")

X_train = adult.drop(['Id', 'income', 'fnlwgt', 'education'], axis=1).copy()

X_train = X_train.apply(preprocessing.LabelEncoder().fit_transform)

X_train = preprocessing.StandardScaler().fit_transform(X_train)

evaluate_model(X_train, Y_train, 3)
X_train = adult.drop(['Id', 'income', 'fnlwgt', 'education'], axis=1).copy()

X_train_preprocessed = X_train.apply(preprocessing.LabelEncoder().fit_transform)

X_train_preprocessed = preprocessing.StandardScaler().fit_transform(X_train_preprocessed)
X_train.describe(include='all')
Y_train = adult['income']
Y_train.describe()
def avalia_k(lista_k):

  resultados = []

  for i in range(len(lista_k)):

    k = lista_k[i]

    print(f"[{i+1}/{len(lista_k)}] Treinando KNN para K={k}")

    knn = KNeighborsClassifier(n_neighbors=k)

    score = cross_val_score(knn, X_train_preprocessed, Y_train, cv=10)

    media = score.mean()

    print(f"\tAcurácia média: {media}")

    resultados.append([k, score.mean()])

  return resultados
def mostra_acuracia_k(acuracia_k):

  acuracia_k.plot()

  acuracia_k.plot(style='.', color='blue')

  plt.grid()

  plt.xlabel("K")

  plt.ylabel("Acurácia")

  plt.title("Evolução da acurácia com o K")
lista_k = [1, 10, 20, 30, 40, 50, 60]

resultados = avalia_k(lista_k)
resultados = np.array(resultados)

acuracia_k = pd.Series(resultados[:,1], index=resultados[:,0])
mostra_acuracia_k(acuracia_k)
mostra_acuracia_k(acuracia_k[10:])
lista_k = [25, 35]

resultados = avalia_k(lista_k)

resultados = np.array(resultados)
acuracia_k = pd.concat([acuracia_k, pd.Series(resultados[:,1], index=resultados[:,0])])
acuracia_k = acuracia_k.sort_index()
mostra_acuracia_k(acuracia_k)
mostra_acuracia_k(acuracia_k[20:40])
lista_k = [21, 22, 23, 24, 26, 27, 28, 29]

resultados = avalia_k(lista_k)

resultados = np.array(resultados)
acuracia_k = pd.concat([acuracia_k, pd.Series(resultados[:,1], index=resultados[:,0])])
acuracia_k = acuracia_k.sort_index()
mostra_acuracia_k(acuracia_k)
mostra_acuracia_k(acuracia_k[20:35])
lista_k = [31, 32, 33, 34, 36, 37, 38, 39]

resultados = avalia_k(lista_k)

resultados = np.array(resultados)
acuracia_k = pd.concat([acuracia_k, pd.Series(resultados[:,1], index=resultados[:,0])])

acuracia_k = acuracia_k.sort_index()
mostra_acuracia_k(acuracia_k)
mostra_acuracia_k(acuracia_k[20:40])
evaluate_model(X_train_preprocessed, Y_train, 32)
knn_final = KNeighborsClassifier(n_neighbors=32)

knn_final.fit(X_train_preprocessed, Y_train)
X_test = adult_test.drop(['Id', 'fnlwgt', 'education'], axis=1).copy()

X_test_preprocessed = X_test.apply(preprocessing.LabelEncoder().fit_transform)

X_test_preprocessed = preprocessing.StandardScaler().fit_transform(X_test_preprocessed)
Y_predicted = knn_final.predict(X_test_preprocessed)
submission = pd.Series(Y_predicted)

submission
submission.to_csv("submission.csv",

                  header=["income"],

                  index_label="Id")