import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

df_treino = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", na_values = '?')

df_treino.set_index('Id',inplace=True)



df_teste = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", na_values = '?')

df_teste.set_index('Id',inplace=True)
print('Forma do DataFrame:', df_treino.shape)

df_treino.info()



df_treino.rename(columns={'marital.status': 'marital_status', 'native.country': 'native_country'}, inplace=True)

df_teste.rename(columns={'marital.status': 'marital_status', 'native.country': 'native_country'}, inplace=True)



df_treino.head()

df_treino.isna().sum()
df_treino['income'] = LabelEncoder().fit_transform(df_treino['income'])
plt.figure(figsize=(10,8))

sns.heatmap(df_treino.corr(), vmin=-1, vmax=1, annot=True)

plt.show()
df_treino['age'].hist()

plt.xlabel("Idade")

plt.ylabel("Frequência")

plt.title("Distribuição da idade")

df_treino[df_treino['income'] == 0]['age'].hist(label="<=50K")

df_treino[df_treino['income'] == 1]['age'].hist(label=">50K")

plt.xlabel("Idade")

plt.ylabel("Frequência")

plt.title("Discriminação da idade por faixa de ganhos")

df_treino['workclass'].value_counts().plot(kind='pie')

plt.title("Distribuição Workclass")
df_treino['workclass'] = df_treino['workclass'].fillna(df_treino.mode().iloc[0].loc['workclass'])

df_teste['workclass'] = df_teste['workclass'].fillna(df_teste.mode().iloc[0].loc['workclass'])



df_treino.isna().sum()
sns.catplot(y="workclass", x="income", kind="bar", data=df_treino)
df_treino = df_treino.drop('fnlwgt', axis=1)

df_teste = df_teste.drop('fnlwgt', axis=1)
sns.catplot(y="education", x="education.num", kind="bar", data=df_treino)

df_treino = df_treino.drop('education', axis=1)

df_teste = df_teste.drop('education', axis=1)
df_treino['education.num'].hist()

plt.xlabel("Anos de Escola")

plt.ylabel("Frequência")

plt.title("Distribuição da Escolaridade")

df_treino[df_treino['income'] == 0]['education.num'].hist(label="<=50K")

df_treino[df_treino['income'] == 1]['education.num'].hist(label=">50K")

plt.xlabel("Escolaridade")

plt.ylabel("Frequência")

plt.title("Discriminação da escolaridade por faixa de ganhos")
df_treino['marital_status'].value_counts().plot(kind='pie')

plt.title("Distribuição Marital Status")
sns.catplot(y="marital_status", x="income", kind="bar", data=df_treino)
df_treino['occupation'].value_counts().plot(kind='pie')

plt.title("Distribuição Occupation")
df_treino['occupation'] = df_treino['occupation'].fillna("Unknown")

df_treino.isna().sum()



df_teste['occupation'] = df_teste['occupation'].fillna("Unknown")

sns.catplot(y="occupation", x="income", kind="bar", data=df_treino)
df_treino['relationship'].value_counts().plot(kind='pie')

plt.title("Distribuição Relationship")
sns.catplot(y="relationship", x="income", kind="bar", data=df_treino)
df_treino['race'].value_counts().plot(kind='pie')

plt.title("Distribuição Race")
sns.catplot(y="race", x="income", kind="bar", data=df_treino)
df_treino['sex'].value_counts().plot(kind='pie')

plt.title("Distribuição Sex")
sns.catplot(y="sex", x="income", kind="bar", data=df_treino)
df_treino['capital.gain'].hist()

plt.xlabel("Capital Gain")

plt.ylabel("Frequência")

plt.title("Distribuição de Capital Gain")

df_treino[df_treino['income'] == 0]['capital.gain'].hist(label="<=50K")

df_treino[df_treino['income'] == 1]['capital.gain'].hist(label=">50K")

plt.xlabel("Capital Gain")

plt.ylabel("Frequência")

plt.title("Discriminação de Capital Gain por faixa de ganhos")
df_treino['capital.loss'].hist()

plt.xlabel("Capital Loss")

plt.ylabel("Frequência")

plt.title("Distribuição de Capital Loss")
df_treino[df_treino['income'] == 0]['capital.loss'].hist(label="<=50K")

df_treino[df_treino['income'] == 1]['capital.loss'].hist(label=">50K")

plt.xlabel("Capital Loss")

plt.ylabel("Frequência")

plt.title("Discriminação de Capital Loss por faixa de ganhos")
df_treino['hours.per.week'].hist()

plt.xlabel("Hours per Week")

plt.ylabel("Frequência")

plt.title("Distribuição de Hours per Week")
df_treino[df_treino['income'] == 0]['hours.per.week'].hist(label="<=50K")

df_treino[df_treino['income'] == 1]['hours.per.week'].hist(label=">50K")

plt.xlabel("Hours per Week")

plt.ylabel("Frequência")

plt.title("Discriminação de Hours per Week por faixa de ganhos")
df_treino['native_country'].value_counts().plot(kind='pie')

plt.title("Distribuição Native Country")
df_treino = df_treino.drop(['native_country'], axis = 1) 

df_treino.info()



df_teste = df_teste.drop(['native_country'], axis = 1) 

df_teste.info()
workclass_dummies = pd.get_dummies(df_treino.workclass)

marital_status_dummies = pd.get_dummies(df_treino.marital_status)

occupation_dummies = pd.get_dummies(df_treino.occupation)

relationship_dummies = pd.get_dummies(df_treino.relationship)

race_dummies = pd.get_dummies(df_treino.race)

sex_dummies = pd.get_dummies(df_treino.sex)



df_treino = pd.concat([df_treino, workclass_dummies, marital_status_dummies, occupation_dummies ,relationship_dummies, race_dummies,sex_dummies], axis=1)



df_treino = df_treino.drop(['workclass', 'marital_status', 'occupation', 'relationship', 'race','sex'], axis = 1)



workclass_dummies = pd.get_dummies(df_teste.workclass)

marital_status_dummies = pd.get_dummies(df_teste.marital_status)

occupation_dummies = pd.get_dummies(df_teste.occupation)

relationship_dummies = pd.get_dummies(df_teste.relationship)

race_dummies = pd.get_dummies(df_teste.race)

sex_dummies = pd.get_dummies(df_teste.sex)



df_teste = pd.concat([df_teste, workclass_dummies, marital_status_dummies, occupation_dummies ,relationship_dummies, race_dummies,sex_dummies], axis=1)



df_teste = df_teste.drop(['workclass', 'marital_status', 'occupation', 'relationship', 'race','sex'], axis = 1)

Y_treino = np.array(df_treino['income']) 

df_treino= df_treino.drop(['income'], axis = 1) 

X_treino = np.array(df_treino)



X_teste = np.array(df_teste)
best_k = 10

best_acc = 0.0



busca_inicial = [10,15,20,25,30]



print('Iniciando busca : ')

for k in busca_inicial:

    KNN = KNeighborsClassifier(n_neighbors=k)

    acc = cross_val_score(KNN, X_treino, Y_treino, cv=5, scoring='accuracy').mean()

    

    print('Número de vizinhos =', k,': Acurácia',100 * acc)

    

    if acc > best_acc:

        best_acc = acc

        best_k = k

print('********************************************')



print('Examinando os valores próximos de',best_k)



minimo = best_k - 3

maximo = best_k + 3



for k in range(minimo, maximo):

    KNN = KNeighborsClassifier(n_neighbors=k)

    acc = cross_val_score(KNN, X_treino, Y_treino, cv=5, scoring='accuracy').mean()

    

    print('Número de vizinhos =', k,': Acurácia',100 * acc)

    

    if acc > best_acc:

        best_acc = acc

        best_k = k

print('******************************************')



print('Melhor k:', best_k)

print('Melhor acurácia:', 100 * best_acc)
knn_20 = KNeighborsClassifier(n_neighbors=20)

knn_20.fit(X_treino, Y_treino)

Y_teste = knn_20.predict(X_teste)
Y_teste_df = pd.DataFrame()

Y_teste_df[0] = df_teste.index

Y_teste_df[1] = Y_teste

Y_teste_df.columns = ['Id','Income']

Y_teste_df.set_index('Id', inplace=True)



Y_teste_df[Y_teste_df['Income'] == 0] = '<=50K'

Y_teste_df[Y_teste_df['Income'] == 1] = '>50K'

Y_teste_df.to_csv('submission.csv', index = True, index_label = 'Id')