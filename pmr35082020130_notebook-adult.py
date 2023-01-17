import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



%matplotlib inline

plt.style.use('seaborn')
from subprocess import check_output
df_treino= pd.read_csv("../input/adult-pmr3508/train_data.csv", index_col=['Id'], na_values="?")
total = df_treino.isnull().sum().sort_values(ascending = False)

percentual = ((df_treino.isnull().sum()/df_treino.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percentual], axis = 1, keys = ['Total', '%'])

missing_data.head()
print('occupation:\n')

print(df_treino['occupation'].describe())



print('\n\nworkclass:\n')

print(df_treino['workclass'].describe())



print('\n\nnative.country:\n')

print(df_treino['native.country'].describe())
value = df_treino['workclass'].describe().top

df_treino['workclass'] = df_treino['workclass'].fillna(value)



value = df_treino['native.country'].describe().top

df_treino['native.country'] = df_treino['native.country'].fillna(value)



value = df_treino['occupation'].describe().top

df_treino['occupation'] = df_treino['occupation'].fillna(value)
df_treino.shape
df_treino.index
df_treino.head(20)
df_treino.describe()
df_treino.info()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df = df_treino.copy()

df['income'] = le.fit_transform(df['income'])

df['income']
sns.set_style("white")

plt.figure(figsize=(16, 7))

data = np.random.normal(size=(30,30)) + np.arange(30) / 2

sns.boxplot(data=df_treino);
plt.figure(figsize=(10, 7))

df_treino['capital.gain'].hist(color = 'mediumaquamarine')

plt.xlabel('capital gain')

plt.ylabel('quantity')

plt.title('Histograma de Capital Ganho')
plt.figure(figsize=(10, 7))

df_treino['capital.loss'].hist(color = 'mediumaquamarine')

plt.xlabel('capital loss')

plt.ylabel('quantity')

plt.title('Histograma de Capital Perdido')
plt.figure(figsize=(10, 7))

df_treino['age'].hist(color = 'mediumaquamarine')

plt.xlabel('age')

plt.ylabel('quantity')

plt.title('Histograma de Idade')
sns.set()

plt.figure(figsize=(13,7))

sns.distplot(df_treino['age'], color = 'rosybrown', bins = 50)

plt.ylabel('quantity')

plt.title('Distribution da Idade')

plt.figure(figsize=(10, 7))

df_treino['hours.per.week'].hist(color = 'mediumaquamarine')

plt.xlabel('hours per week')

plt.ylabel('quantity')

plt.title('Histograma de Horas Por Semana de Trabalho')
plt.figure(figsize=(5, 7))

df_treino['income'].hist(color = 'mediumaquamarine')

plt.xlabel('income')

plt.ylabel('quantity')

plt.title('Histograma de Renda')
mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))

plt.figure(figsize=(10,10))

sns.heatmap(df.corr(), mask=mask, square = True, annot=True, vmin=-1, vmax=1, cmap='icefire')

plt.show()
sns.catplot(x="income", y="hours.per.week", kind="box", data=df, palette="Accent");

sns.catplot(x="income", y="education.num", kind="box", data=df, palette="PuOr");

sns.catplot(x="income", y="age", kind="box", data=df, palette="PuOr");
sns.catplot(x="income", y="capital.gain", kind="boxen", data=df, palette="Accent");

sns.catplot(x="income", y="capital.gain", data=df, palette="Accent");
sns.catplot(x="income", y="capital.loss", kind="boxen", data=df, palette="PuOr");

sns.catplot(x="income", y="capital.loss", data=df, palette ="PuOr");
sns.countplot(x="income", hue="sex", data=df, color="crimson")

plt.title('Relação entre renda e sexo')

plt.figure(figsize=(10, 7))

df['sex'].value_counts().plot(kind = 'pie')
male = df[df['sex'] == 'Male'].count()[0]

female = df.shape[0] - male



print("Podemos ver que a quantidade de mulheres que recebem >50k por ano é bem mais baixa que a de homens, em torno de " ,female*100//(female+male),"%")
plt.figure(figsize=(16, 6))

sns.barplot(x="income", y="education", hue="sex", data=df, color="crimson")

plt.title('Relação entre Renda, Educação e Sexo')
sns.catplot(y="workclass", x="income", kind="bar", data=df);

sns.countplot(x="income", hue="workclass", data=df)
sns.barplot(x="income", y="workclass", hue="sex", data= df,color="crimson")
sns.catplot(y="race", x="income", kind="bar", data=df, palette="Accent");
sns.countplot(x="income", hue="race", data=df)
sns.barplot(x="income", y="race", hue="sex", data=df, color="crimson")
sns.catplot(y="marital.status", x="income", kind="bar", data=df);
sns.countplot(x="income", hue="marital.status", data=df)
sns.catplot(y="occupation", x="income", kind="bar", height=5, aspect=2, data=df);
sns.countplot(x="income", hue="occupation", data=df)
sns.barplot(x="income", y="occupation", hue="sex", data=df, color="crimson")
sns.catplot(y="native.country", x="income", kind="bar", height=10, aspect=1, data=df)
df["native.country"].value_counts()
df_treino.drop_duplicates(keep='first', inplace=True)
df_treino = df_treino.drop(['fnlwgt', 'native.country'], axis=1)
Y_treino = df_treino.pop('income') #variável target



X_treino = df_treino
#Variáveis Esparsas

spa_cols = ['capital.gain', 'capital.loss']



#Variáveis Numéricas

num_cols = list(X_treino.select_dtypes(include=[np.number]).columns.values) 

num_cols.remove('capital.gain')

num_cols.remove('capital.loss')



#Variáveis Qualitativas

quali_cols = list(X_treino.select_dtypes(exclude=[np.number]).columns.values)

from sklearn.impute import SimpleImputer #Usamos um transformador para preencher missing data usando a estratégia de substitui-lo pela **moda**.

from sklearn.preprocessing import OneHotEncoder #para transformar os dados para quantitativos

from sklearn.pipeline import Pipeline



quali_pipeline = Pipeline(steps = [('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(drop='if_binary'))])
from sklearn.preprocessing import StandardScaler 

from sklearn.preprocessing import RobustScaler



#transformar todas as variáveis para uma mesma escala

num_pipeline = Pipeline(steps=[('scaler', StandardScaler())])

spa_pipeline = Pipeline(steps=[('scaler', RobustScaler())])
from sklearn.compose import ColumnTransformer



preprocessador = ColumnTransformer(transformers=[('num', num_pipeline, num_cols),('spr', spa_pipeline, spa_cols),('quali', quali_pipeline, quali_cols)])
X_treino = preprocessador.fit_transform(X_treino)
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier



vizinhos = [5,10,13,14,15,20,25,30,35,40]



for n in vizinhos:

    score = cross_val_score(KNeighborsClassifier(n_neighbors=n), X_treino, Y_treino, cv=13, scoring="accuracy").mean()

    

    print("Número de vizinhos", n," | Acurácia do Modelo: ", score)
knn = KNeighborsClassifier(n_neighbors=14)
knn.fit(X_treino, Y_treino)
teste= pd.read_csv("../input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values="?")
X_teste = teste.drop(['fnlwgt', 'native.country'], axis=1)
X_teste = preprocessador.transform(X_teste)
predictions = knn.predict(X_teste)
predictions
submission = pd.DataFrame()

submission[0] = teste.index

submission[1] = predictions

submission.columns = ['Id', 'income']





submission.head()
submission.to_csv('submission.csv', index=False)