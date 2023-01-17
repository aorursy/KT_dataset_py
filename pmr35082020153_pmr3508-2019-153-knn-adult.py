import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import sklearn
adult_train = pd.read_csv("../input/adult-pmr3508/train_data.csv", na_values="?")

adult_train.head()
adult_train.shape
adult_test = pd.read_csv("../input/adult-pmr3508/test_data.csv", na_values="?")

adult_test.shape
total = adult_train.isnull().sum().sort_values(ascending = False)

percent = ((adult_train.isnull().sum()/adult_train.isnull().count())*100).sort_values(ascending = False)

df_dadosFaltantes = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

df_dadosFaltantes.head()
adult_train['occupation'] = adult_train['occupation'].fillna(adult_train['occupation'].describe().top)

adult_train['workclass'] = adult_train['workclass'].fillna(adult_train['workclass'].describe().top)

adult_train['native.country'] = adult_train['native.country'].fillna(adult_train['native.country'].describe().top)
total = adult_train.isnull().sum().sort_values(ascending = False)

percent = ((adult_train.isnull().sum()/adult_train.isnull().count())*100).sort_values(ascending = False)

df_dadosFaltantes = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

df_dadosFaltantes.head()
numFeatures=[adult_train.columns.tolist()[index] for index in [1,3,5,11,12,13]]

label=adult_train.columns.tolist().pop()



sns.set()

sns.pairplot(adult_train, hue = label, vars = numFeatures, diag_kind = 'auto', diag_kws={'bw':"1.0"})
adult_train.describe()
sns.set()

plt.figure(figsize=(13,7))

sns.distplot(adult_train['capital.gain'],kde_kws={'bw':"1.0"})
sns.set()

plt.figure(figsize=(13,7))

sns.distplot(adult_train['capital.loss'],kde_kws={'bw':"1.0"})
plt.figure(figsize=(13,7))

plt.hist([adult_train['capital.gain'], adult_train['capital.loss']])

plt.show()
sns.catplot(x="income", y="capital.gain", height=10, aspect=1, data=adult_train)
sns.catplot(x="income", y="capital.loss", height=10, aspect=1, data=adult_train)
sns.set()

plt.figure(figsize=(13,7))

sns.distplot(adult_train['age'], kde_kws={'bw':"1.0"})
plt.figure(figsize=(13, 7))

adult_train['age'].hist(color = 'red')

plt.xlabel('age')

plt.ylabel('quantity')

plt.title('Age histogram')
sns.catplot(x="income", y="age", height=10, aspect=1, data=adult_train)
sns.set()

plt.figure(figsize=(13,7))

sns.distplot(adult_train['fnlwgt'])
sns.catplot(x="income", y="fnlwgt", height=10, aspect=1, data=adult_train)
sns.set()

plt.figure(figsize=(13,7))

sns.distplot(adult_train['education.num'])
plt.figure(figsize=(13, 7))

adult_train['education.num'].hist(color = 'red')

plt.xlabel('education.num')

plt.ylabel('quantity')

plt.title('Education histogram')
sns.catplot(x="income", y="education.num", height=10, aspect=1, data=adult_train)
sns.catplot(x="income", y="education.num", kind="boxen", height=10, aspect=1, data=adult_train)
sns.set()

plt.figure(figsize=(13, 7))

sns.distplot(adult_train['hours.per.week'])
sns.catplot(x="income", y="hours.per.week", height=10, aspect=1, data=adult_train)
sns.catplot(x="income", y="hours.per.week", kind="boxen", height=10, aspect=1, data=adult_train)
adult_train.head()
aux = pd.get_dummies(adult_train, columns=["income"])

aux.head()
sns.set()

plt.figure(figsize=(13, 7))

corrMatrix = aux.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
aux = pd.get_dummies(adult_train, columns=["income"])

aux.head()
plt.figure(figsize=(13, 7))

adult_train['workclass'].value_counts().plot(kind='bar',color='coral')
adult_train['workclass'].value_counts()
sns.catplot(y="workclass", x="income_>50K", kind="bar", height=10, aspect=1, data=aux)
plt.figure(figsize=(13, 7))

adult_train['marital.status'].value_counts().plot(kind='bar',color='coral')
sns.catplot(y="marital.status", x="income_>50K", kind="bar", height=10, aspect=1, data=aux)
plt.figure(figsize=(13, 7))

adult_train['occupation'].value_counts().plot(kind='bar',color='coral')
sns.catplot(y="occupation", x="income_>50K", kind="bar", height=10, aspect=1, data=aux)
plt.figure(figsize=(13, 7))

adult_train['relationship'].value_counts().plot(kind='bar',color='coral')
sns.catplot(y="relationship", x="income_>50K", kind="bar", height=10, aspect=1, data=aux)
plt.figure(figsize=(13, 7))

adult_train['race'].value_counts().plot(kind='bar',color='coral')
adult_train['race'].value_counts()
sns.catplot(y="race", x="income_>50K", kind="bar", height=10, aspect=1, data=aux)
plt.figure(figsize=(13, 7))

adult_train['sex'].value_counts().plot(kind='bar',color='coral')
sns.catplot(y="sex", x="income_>50K", kind="bar", height=10, aspect=1, data=aux)
plt.figure(figsize=(13, 7))

adult_train['native.country'].value_counts().plot(kind='bar',color='coral')
adult_train['native.country'].value_counts()
print('Porcentagem das naciononalidades do dataframe')

adult_train['native.country'].value_counts()/adult_train['native.country'].value_counts().sum()
sns.catplot(y="native.country", x="income_>50K", kind="bar", height=10, aspect=1, data=aux)
adult_train.drop_duplicates(keep='first', inplace=True)
adult_train = adult_train.drop(['fnlwgt', 'education', 'native.country'], axis=1)

adult_train.head()
label_train = adult_train.pop('income')

features_train = adult_train
label_train.head()
features_train.head()
numFeatures=[features_train.columns.tolist()[index] for index in [1,3,11]]

print(numFeatures)

catFeatures=[features_train.columns.tolist()[index] for index in [2,4,5,6,7,8]]

print(catFeatures)

conFeatures=[features_train.columns.tolist()[index] for index in [9,10]]

print(conFeatures)
features_train = pd.get_dummies(features_train, columns= catFeatures)
features_train.head()
label_train = pd.get_dummies(label_train)
label_train.head()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



numPipe = Pipeline(steps = [

    ('scaler', StandardScaler())

])
from sklearn.preprocessing import RobustScaler



conPipe = Pipeline(steps = [

    ('scaler', RobustScaler())

])
from sklearn.compose import ColumnTransformer



colTransformer = ColumnTransformer(transformers = [

    ('num', numPipe, numFeatures),

    ('spr', conPipe, conFeatures)

])
features_train = colTransformer.fit_transform(features_train)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
scores = [0]*31

n=1



while n<31:

    score = cross_val_score(KNeighborsClassifier(n_neighbors=n), features_train, label_train, cv = 5, scoring="accuracy").mean()    

    scores[n] = score

    n +=1



n = scores.index(max(scores))
print('O número de vizinhos com maior acurácia é:',n)

print('A melhor acurácia de nosso modelo é:', scores[n])
plt.figure(figsize=(13,7))

for n in range(len(scores)):

    plt.scatter(n,scores[n])

plt.ylabel('Acurácia')

plt.xlabel('Número de vizinhos')

plt.title('Acurácia x Número de vizinhos')

plt.show()    
kNN = KNeighborsClassifier(n_neighbors=27)

kNN.fit(features_train, label_train)
features_test = adult_test.drop(['fnlwgt', 'native.country', 'education'], axis=1)
features_test = colTransformer.fit_transform(features_test)
prediction = kNN.predict(features_test)
prediction
lista = []



for item in prediction:

  if item[0] == 1:

    lista.append('<=50K')

  if item[0] == 0:

    lista.append('>50K')



array = np.array(lista)
array
submission = pd.DataFrame()
submission[0] = adult_test.index

submission[1] = array

submission.columns = ['Id','income']
submission.head()
submission.to_csv('submission.csv',index = False)