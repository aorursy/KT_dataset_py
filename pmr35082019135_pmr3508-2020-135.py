# Importando as bibliotecas a serem utilizadas



import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import preprocessing
#Importando os dados de treino

adult = pd.read_csv('/kaggle/input/adult-pmr3508/train_data.csv',

        names=['Age','Workclass', 'Final Weight', 'Education', 'Education-Num', 'Marital Status', 'Occupation',

               'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', "Hours per week", "Country", "Income"],

        sep=r'\s*,\s*',

        engine='python',

        index_col=[0],

        skiprows=[0],

        na_values="?")



adult.shape
# Pré visualisação do Dataset de treino

adult.head()
adult.info()
adult.isnull().mean(axis=0)
nadata_columns = ['Occupation', 'Workclass', 'Country']

for coluna in nadata_columns:

    print('\n\t',coluna, '\n')

    print(adult[coluna].describe())



    moda = adult[coluna].describe().top

    adult[coluna] = adult[coluna].fillna(moda)
adult.isnull().mean(axis=0)
from sklearn.preprocessing import LabelEncoder
num_adult = adult.copy()

# Binarização do rótulo 'Income'

num_adult.Income = pd.get_dummies(num_adult.Income.to_frame(), drop_first=True)



# Discretização para as outras colunas qualitativas 

categorical = num_adult[['Workclass', 'Education', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex',"Country"]]

categorical = categorical.apply(LabelEncoder().fit_transform)

num_adult[['Workclass', 'Education', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex',"Country"]] = categorical



num_adult.head()
adult.describe()
menor50k = adult["Income"].value_counts()['<=50K']

maior50k = adult["Income"].value_counts()['>50K']



values, labels = [menor50k,maior50k], ['Less than 50K', 'More than 50K']

plt.pie(values, labels = labels, autopct='%2.2f%%', radius=1.5, textprops={'fontsize': 14})





plt.show()
# Montagem do DataFrame com os coeficientes de correlação

correlacao = adult.copy()

correlacao.Income = pd.get_dummies(correlacao.Income.to_frame(), drop_first=True)

correlacao = correlacao.corr()['Income'].to_frame()

correlacao
fig, ax = plt.subplots(figsize=(8,6))

ax.grid(lw=.5)

sns.violinplot(x='Age', y='Income', data=adult, linewidth = .9, ax=ax)

plt.show()
neduc = adult.groupby(['Education-Num', 'Income']).size().unstack()

neduc['sum'] = adult.groupby('Education-Num').size()

neduc = neduc.astype('float64')

l = len(neduc.iloc[:])

for i in range(l):

    for j in range(2):

        neduc.iloc[i][j] = neduc.iloc[i][j] / neduc.iloc[i][2]

neduc = neduc.drop(['sum'], axis=1)

neduc.plot(kind = 'bar', stacked = True, figsize= (8,6))

plt.show()
fig, ax = plt.subplots(figsize=(8,6))

ax.grid(lw=.5)

sns.violinplot(x='Hours per week', y='Income', data=adult, linewidth = 1, ax=ax)

plt.show()
fig, axes = plt.subplots(1,2, figsize=(18, 5))

axes[0].grid(lw=.5)

axes[1].grid(lw=.5)

sns.stripplot(ax=axes[0], data=adult, x='Capital Gain', y='Income')

sns.stripplot(ax=axes[1], data=adult, x='Capital Loss', y='Income')

plt.show()
total = adult["Country"].count()

unitedstatians = adult["Country"].value_counts()['United-States']

othernations = total - unitedstatians



values, labels = [unitedstatians,othernations], ['United-States', 'Other-nations']

plt.pie(values, labels = labels, autopct='%2.2f%%', radius=1.5, textprops={'fontsize': 14})



unitedstatians_pct = unitedstatians/total

othernations_pct = 1 - unitedstatians_pct

othernations_num = len(adult["Country"].value_counts())-1



text = '{:.2%}'.format(unitedstatians_pct) + ' dos valores da variável Country são dos Estados Unidos e apenas '

text +='{:.2%}'.format(othernations_pct) + ' se dividem entre outros ' + str(othernations_num) + ' países.'



print(text)

plt.show()
fig, axes = plt.subplots(ncols = 2)

plt.tight_layout()



educ = adult.groupby(['Education', 'Income'])

educ = educ.size().unstack()

educ['sum'] = adult.groupby('Education').size()

educ = educ.astype('float64')

l = len(educ.iloc[:])

for i in range(l):

    for j in range(2):

        educ.iloc[i][j] = educ.iloc[i][j] / educ.iloc[i][2]

educ = educ.drop(['sum'], axis=1)

educ = educ.sort_values('<=50K', ascending = False)[['<=50K', '>50K']]

educ.plot(kind = 'bar', stacked = True, ax = axes[0], figsize= (18,6))

    

neduc = adult.groupby(['Education-Num', 'Income'])

neduc = neduc.size().unstack()

neduc['sum'] = adult.groupby('Education-Num').size()

neduc = neduc.astype('float64')

l = len(neduc.iloc[:])

for i in range(l):

    for j in range(2):

        neduc.iloc[i][j] = neduc.iloc[i][j] / neduc.iloc[i][2]

neduc = neduc.drop(['sum'], axis=1)

neduc = neduc.sort_values('<=50K', ascending = False)[['<=50K', '>50K']]

neduc.plot(kind = 'bar', stacked = True, ax = axes[1], figsize= (15,4))



plt.show()
def plot_vs_income(coluna, posicao):

    sub_df = adult.groupby([coluna, 'Income']).size().unstack()

    sub_df['sum'] = adult.groupby(coluna).size()

    sub_df = sub_df.astype('float64')

    l = len(sub_df.iloc[:])

    for i in range(l):

        for j in range(2):

            sub_df.iloc[i][j] = sub_df.iloc[i][j] / sub_df.iloc[i][2]

    sub_df = sub_df.drop(['sum'], axis=1)

    sub_df = sub_df.sort_values('<=50K', ascending = False)[['<=50K', '>50K']]

    sub_df.plot(kind = 'bar', stacked = True, ax = axes[posicao[0], posicao[1]], figsize= (14,18))



fig, axes = plt.subplots(nrows = 3, ncols = 2)

plt.tight_layout()



categoricos = {'Sex':[0,0], 'Race':[0,1], 'Marital Status':[1,0], 'Relationship':[1,1], 'Workclass':[2,0], 'Occupation':[2,1]}

for i in categoricos:

    plot_vs_income(i, categoricos[i])
adult = adult.drop(['Education', 'Country', 'Final Weight','Relationship'], axis=1)

num_adult = num_adult.drop(['Education', 'Country', 'Final Weight','Relationship'], axis=1)
X_adult = num_adult[['Age','Workclass', 'Education-Num', 'Marital Status', 'Occupation', 'Race',

                     'Sex', 'Capital Gain', 'Capital Loss', "Hours per week"]]

Y_adult = adult.Income
# Importando bibliotecas para aplicação do classificador KNN e da validação cruzada

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
melhor_n, melhor_score = 0, 0

scores = []



# Valores de n considerados para a análise

nn = [10,15,16,17,18,19,20,21,22,23,24,25,30]

print('Busca do melhor número n de vizinhos:')

for n in nn:

    knn = KNeighborsClassifier(n_neighbors=n)

    n_score = np.mean(cross_val_score(knn, X_adult, Y_adult, cv=10))

    print('No. vizinhos:',n,'\t Score:',n_score)

    scores.append(n_score)

    print

    if n_score > melhor_score:

        melhor_score = n_score

        melhor_n = n
plt.plot(nn,scores,'ro')

plt.grid(ls='--')

plt.xlabel('N-Neighbors')

plt.ylabel('Score')

print('Melhor número de vizinhos é', melhor_n, 'com porcentagem de acerto de', melhor_score,'.')
knn = KNeighborsClassifier(n_neighbors = melhor_n)

knn.fit(X_adult,Y_adult)
# Importando dados de teste

adult_test = pd.read_csv('/kaggle/input/adult-pmr3508/test_data.csv',

        names=['Age','Workclass', 'Final Weight', 'Education', 'Education-Num', 'Marital Status', 'Occupation',

               'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', "Hours per week", "Country"],

        sep=r'\s*,\s*',

        engine='python',

        index_col=[0],

        skiprows=[0],

        na_values="?")



adult_test.isnull().mean(axis=0)
# Retiram-se dados faltantes e as variáveis já retiradas da base de treino

adult_test= adult_test.drop(['Education', 'Country', 'Final Weight','Relationship'], axis=1)



# Realiza-se a imputação que foi feita na base de treino

nadata_columns = ['Occupation', 'Workclass']

for coluna in nadata_columns:

    moda = adult_test[coluna].describe().top

    adult_test[coluna] = adult_test[coluna].fillna(moda)
num_adult_test =  adult_test.copy()

categorical = num_adult_test[['Workclass', 'Marital Status', 'Occupation', 'Race', 'Sex']]

categorical = categorical.apply(LabelEncoder().fit_transform)

num_adult_test[['Workclass', 'Marital Status', 'Occupation', 'Race', 'Sex']] = categorical
Xtest_adult = num_adult_test[['Age','Workclass', 'Education-Num', 'Marital Status', 'Occupation', 'Race',

                              'Sex', 'Capital Gain', 'Capital Loss', "Hours per week"]]
YtestPred = knn.predict(Xtest_adult)

YtestPred
predicao = pd.DataFrame()



predicao[0] = adult_test.index

predicao[1] = YtestPred

predicao.columns = ['Id','Income']

predicao.shape
predicao.to_csv('predicao.csv',index = False)