import pandas as pd

import sklearn

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt



from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",index_col=['Id'],engine='python', na_values="?")
adult.head()
adult.drop_duplicates(keep='first', inplace=True)
adult.info()
#Pegando a moda de cada categoria

workclassMode = adult['workclass'].mode()[0]

occupationMode = adult['occupation'].mode()[0]

nativeCountryMode = adult['native.country'].mode()[0]



#Inputando o valor da moda nos valores vazios

adult['workclass'] = adult['workclass'].fillna(workclassMode)

adult['occupation'] = adult['occupation'].fillna(occupationMode)

adult['native.country'] = adult['native.country'].fillna(nativeCountryMode)



adult.info()
#Fazendo uma copia para não Alterar os dados originais

adult_analys=adult.copy()



adult_analys['income'] = LabelEncoder().fit_transform(adult_analys['income'])
plt.figure(figsize=(12, 6))

sns.heatmap(data=adult_analys.corr(), cmap='YlGnBu', linewidths=0.3, annot=True)
sns.catplot(x="sex", col="income", col_wrap=4, data=adult, kind="count")
sns.catplot(x="workclass", col="income", col_wrap=2, data=adult, kind="count", aspect=2)
sns.catplot(x="marital.status", col="income", col_wrap=3, data=adult, kind="count", aspect = 2)
sns.catplot(x="occupation", col="income", col_wrap=1, data=adult, kind="count", aspect=4)
sns.catplot(x="relationship", col="income", col_wrap=2, data=adult, kind="count", aspect=2)
sns.catplot(x="race", col="income", col_wrap=4, data=adult, kind="count", aspect=1.4)
adult['native.country'].value_counts()
sns.catplot(x="income", y="age", kind="violin", data=adult)
sns.catplot(x="income", y="education.num", kind="violin", data=adult)
sns.catplot(x="income", y="capital.gain", kind="violin", data=adult)
sns.catplot(x="income", y="capital.loss", kind="violin", data=adult)
sns.catplot(x="income", y="hours.per.week", kind="violin", data=adult)
choosenFeatures = ["age", "education.num", "marital.status", "relationship","race","sex","capital.gain","income"]



#Transformando os outros dados categoricos em dados numericos para a nossa predição

le = LabelEncoder()

categoricColumn = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']

for column in categoricColumn:

    adult_analys[column] = le.fit_transform(adult_analys[column])





adult_train = adult_analys[choosenFeatures].copy()

adult_train.head()
XAdult_train = adult_train.drop(["income"], axis = 1)

YAdult_train = adult_train["income"]
#Configurações para o GridSearch

kRange = list(range(5, 31))

pOptions = list(range(1,3))

gridParameters = dict(n_neighbors=kRange, p=pOptions)
knnTreino = KNeighborsClassifier(n_neighbors=5)



grid = GridSearchCV(knnTreino, gridParameters, cv=10, scoring='accuracy', n_jobs = -2)  
grid.fit(XAdult_train, YAdult_train)

print(grid.best_estimator_)

print(grid.best_score_)
knnFinal = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=None, n_neighbors=28, p=1, weights='uniform')



knnFinal.fit(XAdult_train, YAdult_train)
import pandas as pd

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
knnFinal = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=None, n_neighbors=28, p=1, weights='uniform')



knnFinal.fit(XAdult_train, YAdult_train)
#Importação da base de teste

adult_Test = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", na_values="?", index_col=['Id'])

adult_Test.shape
adult_Test.info()
#Pegando a moda de cada categoria

workclassMode = adult_Test['workclass'].mode()[0]

occupationMode = adult_Test['occupation'].mode()[0]

nativeCountryMode = adult_Test['native.country'].mode()[0]



#Inputando o valor da moda nos valores vazios

adult_Test['workclass'] = adult_Test['workclass'].fillna(workclassMode)

adult_Test['occupation'] = adult_Test['occupation'].fillna(occupationMode)

adult_Test['native.country'] = adult_Test['native.country'].fillna(nativeCountryMode)



adult_Test.info()
# Instanciando o LabelEncoder

le = LabelEncoder()



categoricColumn = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']



# Transformando as variáveis categóricas em numéricas

for column in categoricColumn:

    adult_Test[column] = le.fit_transform(adult_Test[column])
choosenFeatures = ["age", "education.num", "marital.status", "relationship","race","sex","capital.gain"]



adult_Test = adult_Test[choosenFeatures].copy()

adult_Test.head()
YAdult_Test = knnFinal.predict(adult_Test)
finalArray = []



for i in range(len(YAdult_Test)):

    if (YAdult_Test[i] == 0):

        finalArray.append('<=50K')

    else:

        finalArray.append('>50K')

        

#transformação do array em DataFrame

finalDF = pd.DataFrame({'income': finalArray})
finalDF.to_csv("submission.csv", index = True, index_label = 'Id')