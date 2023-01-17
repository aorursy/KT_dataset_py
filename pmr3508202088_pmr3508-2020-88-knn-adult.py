import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('seaborn')
adult = pd.read_csv('/kaggle/input/adult-pmr3508/train_data.csv',index_col='Id', na_values="?")

adult.shape
adult.head()
adult.info()
adult.describe()
adult.isnull().sum()
adult['workclass'].value_counts().plot(kind = 'bar', title = 'Gráfico de workclass vs número de aparições');
adult['workclass'] = adult['workclass'].fillna(adult['workclass'].mode()[0])
adult['occupation'].value_counts().plot(kind = 'bar', title = 'Gráfico de occupation vs número de aparições');
adult['native.country'].value_counts().plot(kind = 'bar', title = 'Gráfico de native country vs número de aparições');
adult['native.country'] = adult['native.country'].fillna(adult['native.country'].mode()[0])

adult['native.country'].isnull().sum()
adult = adult.dropna()
# Copiando os atributos relevantes

piramide_etaria = adult[['sex', 'age']].copy()

piramide_etaria.head()
# Contagem populacional agrupada por sexo por idade

homens = piramide_etaria[piramide_etaria.sex == 'Male'].groupby('age').count().reset_index()

mulheres = piramide_etaria[piramide_etaria.sex == 'Female'].groupby('age').count().reset_index()



homens.head()
# Inversão de sinal para a população masculina ficar a esquerda do gráfico

homens.sex = homens.sex * (-1)



# Construção do gráfico

plt.figure (figsize = (10,10))

plt.barh(homens.age, homens.sex, label = 'Homens')

plt.barh(mulheres.age, mulheres.sex, label = 'Mulheres')

plt.title('Pirâmide Etária')

plt.xlabel('Número de aparições')

plt.ylabel('Idade')

plt.legend()

plt.show()
for variavel in ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'native.country']: 

    adult[variavel].value_counts().plot(kind = 'bar', title = 'Gráfico de ' + variavel + ' vs número de aparições');

    plt.show()
def comparar_hist(df, var_classe, var_teste):

    

    # Pegar cada classe da variável de classe

    classe = df[var_classe].unique()

    

    # Colocar em um array os sexos de determinada classe (na ondem dos valores da variavel de classe)

    temp = []

    for i in range(len(classe)):

        temp.append(df[df[var_classe] == classe[i]])

        temp[i] = np.array(temp[i][var_teste]).reshape(-1,1)



    fig = plt.figure(figsize=(7,7))

    

    transp = 0.7

    for i in range(len(classe)):

        plt.hist(temp[i], alpha = transp)

    plt.xlabel(var_teste)

    plt.ylabel('Número de aparições')

    plt.title('Histograma de ' + var_teste)

    plt.legend(classe)
comparar_hist(adult, 'income', 'sex')
comparar_hist(adult, 'income', 'age')
comparar_hist(adult, 'income', 'race')
comparar_hist(adult, 'income', 'education.num')
comparar_hist(adult, 'income', 'marital.status')
comparar_hist(adult, 'income', 'hours.per.week')
comparar_hist(adult, 'income', 'capital.gain')

comparar_hist(adult, 'income', 'capital.loss')
comparar_hist(adult, 'income', 'relationship')
comparar_hist(adult, 'income', 'workclass')

comparar_hist(adult, 'income', 'native.country')

comparar_hist(adult, 'income', 'occupation')
# Categóricas que necessitam ser convertidas para numérica

cat_to_num = ['sex', 'race', 'marital.status', 'occupation', 'relationship']



for atribute in cat_to_num:

    adult[atribute] = adult[atribute].astype('category')

    string_num = atribute + '.num'

    adult[string_num] = adult[atribute].cat.codes

    

adult.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
Xtrain = adult[['age', 'education.num', 'hours.per.week', 'sex.num', 'race.num', 'marital.status.num', 'capital.gain', 'capital.loss', 'relationship.num']]

Ytrain = adult.income
# Lista de 10 a 50 como possibilidade de k's

n_vizinhos_teste = list(range(10,50))



# Dicionario para armazenar valores

dict_scores = {}



# Numero de folds na validação cruzada

n_folds = 5



# Iterar para cada valor possível de k (10 a 50)

for n_viz in n_vizinhos_teste:

    # kNN

    knn = KNeighborsClassifier(n_neighbors = n_viz)

    score = cross_val_score(knn, Xtrain, Ytrain, cv = n_folds, scoring = 'accuracy').mean()

    dict_scores[n_viz] = score



best_k = max(dict_scores, key=dict_scores.get)

print('Melhor k: ', best_k, ' | Acuracia: ', dict_scores[best_k])

knn = KNeighborsClassifier(n_neighbors = best_k)

knn.fit(Xtrain, Ytrain)
test = pd.read_csv('/kaggle/input/adult-pmr3508/test_data.csv', index_col = 'Id', na_values = '?')
test.isnull().sum()
# Mesmo tratamento da base de treino

test['workclass'] = test['workclass'].fillna(test['workclass'].mode()[0])

test['native.country'] = test['native.country'].fillna(test['native.country'].mode()[0])
# Categóricas que necessitam ser convertidas para numérica

cat_to_num = ['sex', 'race', 'marital.status', 'occupation', 'relationship']



for atribute in cat_to_num:

    test[atribute] = test[atribute].astype('category')

    string_num = atribute + '.num'

    test[string_num] = test[atribute].cat.codes

    

test.head()
# Criando uma coluna de income

test.income = np.nan



# Separando os atributos da variável de classe

Xtest = test[['age', 'education.num', 'hours.per.week', 'sex.num', 'race.num', 'marital.status.num', 'capital.gain', 'capital.loss', 'relationship.num']]

Ytest = test.income



Ytest_pred = knn.predict(Xtest)
prediction = pd.DataFrame(index = test.index)

prediction['income'] = Ytest_pred

prediction.head()
prediction.to_csv('submittion.csv')