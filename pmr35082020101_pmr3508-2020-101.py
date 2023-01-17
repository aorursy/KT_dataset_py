import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

df_train = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",index_col=['Id'], na_values='?')

df_train.head()
df_train.info()
total = df_train.isnull().sum().sort_values(ascending = False)

porcentagem = ((df_train.isnull().sum()/df_train.isnull().count())*100).sort_values(ascending = False)

missing = pd.concat([total,porcentagem], axis = 1, keys =  ['Total', '%'])

missing.head()



print('occupation:\n')

print(df_train['occupation'].describe())



print('\n\nworkclass:\n')

print(df_train['workclass'].describe())



print('\n\nnative.country:\n')

print(df_train['native.country'].describe())

#cópia do dataframe original para modificações



df_copy = df_train

from sklearn import preprocessing as prep



#transformando a variável de classe "income" em 0s e 1s

le = prep.LabelEncoder()

df_copy["income"] = le.fit_transform(df_copy["income"])

#Visualização dos dados



plt.figure(figsize = (10,10))

sns.heatmap(df_copy.corr(), square = True, annot = True)

plt.show()
#vamos avaliar as relações das features numéricas com a variável de classe



plt.figure(figsize = (20,20))

sns.catplot(x="income", y="hours.per.week", kind = "boxen", data=df_copy);

plt.figure(figsize = (20,20))

sns.catplot(x="income", y="age", kind = "boxen", data=df_copy);

plt.figure(figsize = (20,20))

sns.catplot(x="income", y="education.num", kind = "boxen", data=df_copy);
plt.figure(figsize = (20,20))

sns.catplot(x="income", y="capital.gain", kind = "boxen", data=df_copy);
plt.figure(figsize = (20,20))

sns.catplot(x="income", y="capital.loss", kind = "boxen", data=df_copy);
#Os 2 últimos boxen plots não trazem muitas informações úteis, vamos utilizar novamente os scatter plots



plt.figure(figsize = (20,20))

sns.catplot(x="income", y="capital.gain", data=df_copy);



plt.figure(figsize = (20,20))

sns.catplot(x="income", y="capital.loss", data=df_copy);

df_train.describe()

#percebemos que essas duas últimas variáveis vistas nos gráficos são bem esparsas e com outliers
#Vamos verificar se há desigualdades nas features qualitativas, de forma a perceber se a feature é relevante para a variável de classe

sns.catplot(x='income', y = 'sex', kind = 'bar', data = df_copy)



sns.catplot(x='income', y = 'race', kind = 'bar', data = df_copy)

sns.catplot(x='income', y = 'workclass', kind = 'bar', data = df_copy)
sns.catplot(x='income', y = 'occupation', kind = 'bar', data = df_copy)
sns.catplot(x='income', y = 'marital.status', kind = 'bar', data = df_copy)
sns.catplot(x='income', y = 'native.country', kind = 'bar', data = df_copy)

sns.catplot(x='income', y = 'relationship', kind = 'bar', data = df_copy)
#Preenchendo os valores faltantes com a moda



value = df_copy['workclass'].describe().top

df_copy['workclass'] = df_copy['workclass'].fillna(value)

value = df_copy['occupation'].describe().top

df_copy['occupation'] = df_copy['occupation'].fillna(value)



#Começo da limpeza de dados

#Removendo duplicatas

df_copy.drop_duplicates(keep= 'first', inplace = True)

base = df_copy.drop(['fnlwgt', 'native.country', 'education'], axis = 1)



#Separando a variável de classe

base_y = base.pop('income')

base_x = base

hot = prep.OneHotEncoder()

scaler = prep.StandardScaler()

robust = prep.RobustScaler()



#vamos dividir em colunas numéricas, esparsas e qualitativas (categóricas)



num = list(base_x.select_dtypes(include=[np.number]).columns.values)

num.remove('capital.loss')

num.remove('capital.gain')

qua = list(base_x.select_dtypes(exclude = [np.number]).columns.values)

spr = ['capital.loss', 'capital.gain']

print(num)

print(qua)

print(spr)
#Aplicando os trabalhos de preparação à base



from sklearn.compose import ColumnTransformer



preprocessor = ColumnTransformer(transformers = [('cat', hot, qua),

                            ('num', scaler, num),

                            ('spr', robust, spr)

])



base_x = preprocessor.fit_transform(base_x)
#Encontrando o melhor hiperparâmetro





from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



score_mean = []

score_std = []



max_mean = 0

max_std = 0

k_max = 0

i = 0

n = [15,20,25,26,27,28,29,30,31,32,33,34,35]

for k in n:

    

    knn = KNeighborsClassifier(n_neighbors = k, p = 2)

    score = cross_val_score(knn,base_x, base_y, cv = 5)

    

    score_mean.append(score.mean())

    score_std.append(score.std())

    

    if score_mean[i] > max_mean:

        k_max = k

        max_mean = score_mean[i]

        max_std = score_std[i]

    i+=1

    

print('Best k is: {0}  | CV accuracy: {1:2.2f}% +/- {2:3.2f}%'.format(k_max,max_mean*100,max_std*100))
knn = KNeighborsClassifier(n_neighbors = k_max, p = 2)

knn.fit(base_x, base_y)
test_data = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values= '?')

X = test_data.drop(['fnlwgt', 'native.country', 'education'], axis = 1)

X.info()

value = X['occupation'].describe().top

X['occupation'] = X['occupation'].fillna(value)

value = X['workclass'].describe().top

X['workclass'] = X['workclass'].fillna(value)

X = preprocessor.transform(X)



predictions = knn.predict(X)

predictions

predictions = le.inverse_transform(predictions)

predictions
submission = pd.DataFrame()

submission[0] = test_data.index

submission[1] = predictions

submission.columns = ['Id','income']

submission.to_csv('submission.csv',index = False)