# importar bibliotecas

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor

%matplotlib inline

plt.style.use('seaborn')
# importar dados de treino

df_train = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv")

df_train.replace('?', np.nan, inplace = True)

df_train.set_index('Id', inplace = True)

df_train.head()
# formato do DataFrame

print('O data frame de teste tem ' , df_train.shape[0], " linhas e ", df_train.shape[1], " colunas.")
# dados faltantes

# Série com o número de dados faltantes em cada coluna

total = df_train.isna().sum().sort_values(ascending = False)

# Série com a porcentagem de dados faltantes sobre o total de dados

percent = ((df_train.isna().sum()/df_train.shape[0])*100).sort_values(ascending = False)

# Criando tabela que mostra dados faltantes e a porcentagem

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
df_train['workclass'] = df_train['workclass'].fillna(df_train['workclass'].describe().top)

df_train['native.country'] = df_train['native.country'].fillna(df_train['native.country'].describe().top)

df_train['occupation'] = df_train['occupation'].fillna(df_train['occupation'].describe().top)

total = df_train.isna().sum().sort_values(ascending = False)

total
#Plotando uma matriz de dispersão



cols = ['age', 'fnlwgt', 'education.num', 'hours.per.week']

sns.set()

sns.pairplot(df_train, vars = cols, hue = 'income')
df_train['age'].plot(kind='hist', bins = 80)

plt.xlabel('Age', size = 20)

plt.ylabel('Frequency' ,size = 20)

plt.title('Age distribution', size = 25)
df_train['sex'].value_counts().plot(kind='pie')

plt.title('Sex', size = 25)

plt.ylabel('')
df_train['race'].value_counts().plot(kind='pie')

plt.title('Race', size = 25)

plt.ylabel('')
df_train['marital.status'].value_counts().plot(kind='bar')

plt.ylabel('Frequency' ,size = 20)

plt.title('Marital Status', size = 25)
#Criar uma tabela com a proporção entre homens e mulheres em relação à renda

income_sex = pd.crosstab(df_train['income'], df_train['sex'], margins = True)

income_sex.drop('All', inplace = True)

income_sex =  round(income_sex*100/income_sex.sum(),2)

income_sex
# criar o gráfico para vizualizar os dados

income_sex.plot(kind='bar')

plt.xlabel('Income', size = 15)

plt.ylabel('Rate (%)', size = 15)

plt.title('Male vs. Female Income', size =20)
income_race = pd.crosstab(df_train['income'], df_train['race'], margins = True)

income_race.drop('All', inplace = True)

income_race =  round(income_race*100/income_race.sum(),2)

income_race
# criar o gráfico para vizualizar os dados

income_race.plot(kind='bar')

plt.xlabel('Income', size = 15)

plt.ylabel('Rate (%)', size = 15)

plt.title('Income for each race', size =20)
income_marital_status = pd.crosstab(df_train['income'], df_train['marital.status'], margins = True)

income_marital_status.drop('All', inplace = True)

income_marital_status =  round(income_marital_status*100/income_marital_status.sum(),2)

income_marital_status
# criar o gráfico para vizualizar os dados

income_marital_status.plot(kind='bar')

plt.xlabel('Income', size = 15)

plt.ylabel('Rate (%)', size = 15)

plt.title('Income for marital status', size =20)
income_education = pd.crosstab(df_train['income'], df_train['education'], margins = True)

income_education.drop('All', inplace = True)

income_education =  round(income_education*100/income_education.sum(),2)

income_education = income_education[['HS-grad','Some-college','Bachelors', 'Masters', 'Doctorate']]

income_education
# criar o gráfico para vizualizar os dados

income_education.plot(kind='bar')

plt.xlabel('Income', size = 15)

plt.ylabel('Rate (%)', size = 15)

plt.title('Income for education', size =20)
Xtrain = df_train[["age","workclass","education.num","marital.status", "occupation","relationship","race","sex","capital.gain", "capital.loss", "hours.per.week"]]

preproc = preprocessing.LabelEncoder()

Xtrain = Xtrain.apply(preproc.fit_transform)
Ytrain = df_train.income
scores_k =[]

for k in [1,3,7,15,30,50]:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, Xtrain, Ytrain, cv=4)

    scores_k.append([k,np.mean(scores)])

scores_k
scores_k =[]

for k in range(25,35):

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, Xtrain, Ytrain, cv=4)

    scores_k.append([k,np.mean(scores)])

scores_k
# importar dados de teste

df_test = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv")

df_test.replace('?', np.nan, inplace = True)

df_test.set_index('Id', inplace = True)

df_test.head()
df_test['workclass'] = df_test['workclass'].fillna(df_test['workclass'].describe().top)

df_test['native.country'] = df_test['native.country'].fillna(df_test['native.country'].describe().top)

df_test['occupation'] = df_test['occupation'].fillna(df_test['occupation'].describe().top)

Xtest = df_test[["age","workclass","education.num","marital.status", "occupation","relationship","race","sex","capital.gain", "capital.loss", "hours.per.week"]]

preproc = preprocessing.LabelEncoder()

Xtestproc = Xtest.apply(preproc.fit_transform)
knn = KNeighborsClassifier(n_neighbors = 30)

knn.fit(Xtrain, Ytrain)
Ypred = knn.predict(Xtestproc)
Ypred.shape
df_test['Income'] = Ypred

df_test
sub = pd.DataFrame()

sub[0] = df_test.index

sub[1] = Ypred

sub.columns = ['Id', 'income']
sub
sub.to_csv('submission.csv', index=False)