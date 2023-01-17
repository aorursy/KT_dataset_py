import pandas as pd

import numpy as np

import seaborn as sns

from sklearn import preprocessing

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

adult_train = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        names=[

        "Id","Age", "Workclass", "fnlwgt", "Education", "education-num", "Martial Status",

        "Occupation", "relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        skiprows=1,

        engine='python',

        na_values="?")



adult_test =  pd.read_csv("../input/adult-pmr3508/test_data.csv",

        names=[

        "Id","Age", "Workclass", "fnlwgt", "Education", "education-num", "Martial Status",

        "Occupation", "relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"],

        sep=r'\s*,\s*',

        skiprows=1,

        engine='python',

        na_values="?")

adult_train.head()
print('Formato do DataFrame: ', adult_train.shape)
adult_train.info()
Nadult_train = adult_train.dropna()

Nadult_test = adult_test.dropna()
Nadult_train.apply(preprocessing.LabelEncoder().fit_transform).corr()
Nadult_train_corr = Nadult_train.apply(preprocessing.LabelEncoder().fit_transform).corr()

sns.heatmap(Nadult_train_corr, vmin=-0.4, vmax=0.4, center=0, cmap='Greens', linewidths=2, linecolor='black')
adult_train['Age'].hist(color = 'blue', grid=False, label = True)

plt.xlabel('Age')
adult_train['Workclass'].describe()
adult_train['Workclass'].value_counts()
adult_train['fnlwgt'].value_counts()
adult_train['fnlwgt'].describe()
menor50K = adult_train[adult_train['Target'] == '<=50K']

maior50K = adult_train[adult_train['Target'] == '>50K']



menor50K['education-num'].hist(color = 'red', alpha = 0.7, grid = False, label = '<=50K')

maior50K['education-num'].hist(color = 'green', alpha = 0.7, grid = False, label = '>50K')



plt.legend()

plt.xlabel('Nível Educacional')
adult_train['Occupation'].value_counts()
menor50K['Occupation'].value_counts().plot(kind = 'bar', color = 'orange')

plt.title('Histograma das ocupações de pessoas com renda anual inferior à 50 mil dólares')
maior50K['Occupation'].value_counts().plot(kind = 'bar', color = 'purple')

plt.title('Histograma das ocupações de pessoas com renda anual superior à 50 mil dólares')
adult_train['Race'].value_counts()
menor50K['Race'].hist(color = 'red', alpha = 0.7, grid = False, label = '<=50K')

maior50K['Race'].hist(color = 'green', alpha = 0.7, grid = False, label = '>50K')
adult_train['Sex'].value_counts()
menor50K['Sex'].value_counts().plot(kind='pie')

plt.legend()
maior50K['Sex'].value_counts().plot(kind='pie')

plt.legend()
male = adult_train[adult_train['Sex'] == 'Male']

female = adult_train[adult_train['Sex'] == 'Female']



male['Hours per week'].hist(color = 'blue', alpha = 0.7, grid = False, label = 'Male')

female['Hours per week'].hist(color = 'cyan', alpha = 0.5, grid = False, label = 'Female')



plt.legend()

plt.xlabel('Horas semanais trabalhadas')
menor50K = adult_train[adult_train['Target'] == '<=50K']

maior50K = adult_train[adult_train['Target'] == '>50K']



menor50K['Hours per week'].hist(color = 'red', alpha = 0.7, grid = False, label = '<=50K')

maior50K['Hours per week'].hist(color = 'green', alpha = 0.7, grid = False, label = '>50K')



plt.legend()

plt.xlabel('Horas semanais trabalhadas')
menor50K = adult_train[adult_train['Target'] == '<=50K']

maior50K = adult_train[adult_train['Target'] == '>50K']



menor50K['Capital Gain'].hist(color = 'red', alpha = 0.7, grid = False, label = '<=50K')

maior50K['Capital Gain'].hist(color = 'green', alpha = 0.7, grid = False, label = '>50K')



plt.legend()

plt.xlabel('Ganho de capital')
menor50K = adult_train[adult_train['Target'] == '<=50K']

maior50K = adult_train[adult_train['Target'] == '>50K']



menor50K['Capital Loss'].hist(color = 'red', alpha = 0.7, grid = False, label = '<=50K')

maior50K['Capital Loss'].hist(color = 'green', alpha = 0.7, grid = False, label = '>50K')



plt.legend()

plt.xlabel('Perda de capital')
adult_train['Country'].value_counts()
Adult_train_final = Nadult_train[['Age', 'education-num', 'Martial Status', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours per week']]

Adult_test_final= adult_test[['Age', 'education-num', 'Martial Status', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours per week']]

Adult_train_final.head()

numAdult_train_final = Adult_train_final.apply(preprocessing.LabelEncoder().fit_transform)

numAdult_test_final = Adult_test_final.apply(preprocessing.LabelEncoder().fit_transform)



X = numAdult_train_final

Y = Nadult_train['Target']

X_test = numAdult_test_final
BestK = 15

BestAcuraccy = 0.0



for k in range(10, 35):

    acuracia = cross_val_score(KNeighborsClassifier(n_neighbors=k), X, Y, cv=10, scoring="accuracy").mean()

    print('k = ', k, ' ;  Acurácia: ', acuracia)

    print('\n')

    if acuracia > BestAcuraccy:

        BestAcuraccy = acuracia

        BestK = k

        

print('O k com maior acurácia é: ', BestK)

print('Melhor acurácia: ', BestAcuraccy)
cknn = KNeighborsClassifier(n_neighbors=25)

cknn.fit(X, Y)
prediction = cknn.predict(X_test)

prediction
submission = pd.DataFrame(prediction, columns=['Income'])

submission.head()
submission.shape
submission.to_csv('submission.csv', index_label = 'Id')
