import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
# Importa os dados de treino

TRAINadult = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        index_col=['Id'],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")



# Importa os dados de teste

TESTadult = pd.read_csv("../input/adult-pmr3508/test_data.csv",

        index_col=['Id'],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")



# Remove os dados faltantes dos dados de treino

nTRAINadult = TRAINadult.dropna()



# Comparação do número de dados com e sem os dados faltantes

print('Treino - Com dados faltantes: ', TRAINadult.shape[0], '; Sem dados faltantes: ', nTRAINadult.shape[0])

nTRAINadult.head()
nTRAINadult.shape
nTRAINadult.describe()
sns.boxplot(x=nTRAINadult["age"])
sns.boxplot(x=nTRAINadult["hours.per.week"])
plt.figure(figsize=(15, 7))

nTRAINadult.groupby('income').sex.hist()

plt.legend(['<=50k','>50k'])

plt.title("Histogram of 'sex' by 'income'")

plt.xlabel('Sex')

plt.ylabel('Number of occurrences')
nTRAINadult['sex'].value_counts()
plt.figure(figsize=(15, 7))

nTRAINadult.groupby('income').age.hist()

plt.legend(['<=50k','>50k'])

plt.title("Histogram of 'age' by 'income'")

plt.xlabel('Age')

plt.ylabel('Number of occurrences')
plt.figure(figsize=(15, 7))

nTRAINadult.groupby('income').race.hist()

plt.legend(['<=50k','>50k'])

plt.title("Histogram of 'race' by 'income'")

plt.xlabel('Race')

plt.ylabel('Number of occurrences')
plt.figure(figsize=(15, 7))

nTRAINadult.groupby('income').relationship.hist()

plt.legend(['<=50k','>50k'])

plt.title("Histogram of 'relationship' by 'income'")

plt.xlabel('Relationship')

plt.ylabel('Number of occurrences')
nTRAINadult.columns
plt.figure(figsize=(15, 7))

plt.title('Native country distribuition in TRAIN set')

nTRAINadult["native.country"].value_counts().plot(kind="pie")
drop_columns = ['fnlwgt','education','native.country']

#Retira as colunas indesejadas na base de treino

baseTRAIN = nTRAINadult.drop(drop_columns, axis = 1)

baseTRAIN.head()
TESTadult.isnull().sum(axis=0)
TESTadult["workclass"].value_counts().plot(kind="pie")

plt.title('Workclass distribuition in TEST set')
TESTadult["occupation"].value_counts().plot(kind="pie")

plt.title('Occupation distribuition in TEST set')
#completa worclass com a moda

TESTadult['workclass'] = TESTadult['workclass'].fillna(TESTadult['workclass'].describe().top)



#completa occupation com a moda

TESTadult['occupation'] = TESTadult['occupation'].fillna(method='pad')
#Conferindo que os dados faltantes em 'workclass' e 'occupation' foram retirados

TESTadult.isnull().sum(axis=0)
TESTadult.shape
#removendo as colunas não relevantes da base de testes

#Retira as colunas indesejadas na base de treino

baseTEST = TESTadult.drop(drop_columns, axis = 1)

baseTEST.head()
baseTEST.shape
#seleciona dados numéricos da base de treino

Xadult = baseTRAIN[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]

Yadult = baseTRAIN.income



#seleciona dados numéricos da base de teste

Xadult_test = baseTEST[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
# Vamos varrer K para encontrar o melhor classificador



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score





scr_mean_max = 0

K_max_num = 0

mean_scores = 0



for K in range(1,51):

    knn = KNeighborsClassifier(n_neighbors=K)

    scores = cross_val_score(knn, Xadult, Yadult,cv=10) # Faz validação cruzada com 10 folds

    mean_scores = scores.mean()

    if ( mean_scores > scr_mean_max):

        scr_mean_max = mean_scores

        K_max_num = K



print("Max score medio: ", scr_mean_max) 

print("'K' do  score máximo: ", K_max_num)

    
print('Com o N_max = ', K_max_num)

knn = KNeighborsClassifier(n_neighbors=K_max_num)

scores_best = cross_val_score(knn, Xadult, Yadult,cv=10) # Fazendo CV novamente para atualizar os valores de score

mean_scores_best_num = scores_best.mean()

print('Novo melhor score: ', mean_scores_best_num)



knn.fit(Xadult, Yadult)



# Predição:



predict_num = knn.predict(Xadult_test)
predict_num
from sklearn import preprocessing



totAdult_test = baseTEST.apply(preprocessing.LabelEncoder().fit_transform) #teste



#para dados de treino

Xadult_tot = baseTRAIN.iloc[:,0:11].apply(preprocessing.LabelEncoder().fit_transform) #treino

Yadult_tot = baseTRAIN.income



Xadult_tot.head() #mostra a base Xadult_tot
#para dados de teste

Xadult_tot_test = totAdult_test.iloc[:,0:11]

Xadult_tot_test.head()
scr_mean_max = 0

K_max_tot = 0

mean_scores = 0



for K in range(1,51):

    knn = KNeighborsClassifier(n_neighbors=K)

    scores = cross_val_score(knn, Xadult_tot, Yadult_tot,cv=10) # Faz validação cruzada com 10 folds

    mean_scores = scores.mean()

    if ( mean_scores > scr_mean_max):

        scr_mean_max = mean_scores

        K_max_tot = K



print("Max score medio: ", scr_mean_max) 

print("'K' do  score máximo: ", K_max_tot)
print('Com o K_max = ', K_max_tot)

knn = KNeighborsClassifier(n_neighbors=K_max_tot)

scores_best = cross_val_score(knn, Xadult_tot, Yadult_tot,cv=10) # Fazendo CV novamente para atualizar os valores de score

mean_scores_best_tot = scores_best.mean()

print('Novo melhor score: ', mean_scores_best_tot)



knn.fit(Xadult_tot, Yadult_tot)



# Predição:



predict_tot = knn.predict(Xadult_tot_test)
predict_tot
print("Dados numéricos: melhor K=", K_max_num, "melhor score medio=", mean_scores_best_num)

print("Numéricos e não-numéricos: melhor K=", K_max_tot, "melhor score medio=", mean_scores_best_tot)
out = pd.DataFrame(predict_tot,columns=['income'])

out.to_csv("PMR3508-2020-76_out.csv", index_label="Id")

out.shape