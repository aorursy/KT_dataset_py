import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import sklearn

import time



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split



%matplotlib inline
# Importando os dados de Treino

# Algumas colunas foram renomeadas para facilitar a leitura e interpretação dos dados durante o processo



dTreino = pd.read_csv ("/kaggle/input/adult-pmr3508/train_data.csv", 

                            engine = 'python',

                            na_values = '?')

dTreino.set_index('Id',inplace=True)

dTreino.rename(columns={'fnlwgt':'total.people',

                            'education.num':'education.months'}, inplace=True)







# Importando os dados de Teste

dTeste = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

                          engine = 'python',

                          na_values = '?')

dTeste.set_index('Id',inplace=True)

dTeste.rename(columns={'fnlwgt':'total.people',

                            'education.num':'education.months'}, inplace=True)



print("Tamanho do DataSet Treino (linhas, colunas):", dTreino.shape)

print("Tamanho do DataSet Teste (linhas, colunas):", dTeste.shape)
dTreino.head()
dTeste.head()
dTreino.info()
dTreino.describe(include='all')
plt.hist(dTreino['age'])
dTreino['sex'].value_counts().plot(kind='pie', autopct='%1.0f%%')
dTreino['race'].hist(figsize=(10,6))
dTreino['native.country'].hist(bins=50, figsize=(15,6), align='mid')

plt.xticks(rotation=90)

plt.show()
dTreino.loc[dTreino['native.country'] != 'United-States', 'native.country'] = 'Other'

dTeste.loc[dTeste['native.country'] != 'United-States', 'native.country'] = 'Other'

dTreino['native.country']
#Dados de Treino

dTreinoFinal = pd.get_dummies(dTreino, columns=['workclass',

                                           'education',

                                           'marital.status',

                                           'occupation',

                                           'relationship',

                                           'race',

                                           'sex',

                                           'native.country'])



#Debug

print('Tamanho (linhas, colunas)=', dTreinoFinal.shape)
#Dados de Teste

dTesteFinal = pd.get_dummies(dTeste, columns=['workclass',

                                           'education',

                                           'marital.status',

                                           'occupation',

                                           'relationship',

                                           'race',

                                           'sex',

                                           'native.country'])



#Debug

print('Tamanho (linhas, colunas)=', dTesteFinal.shape)
dTreinoFinal.describe(include='all')
graficos = dTreino.hist(bins=50, figsize=(50,50), xlabelsize=30, ylabelsize=30)

for x in graficos.ravel():

    x.title.set_size(40)
yTreino = dTreinoFinal['income']



#Debug

print('Tamanho (linhas,colunas)=', yTreino.shape)

yTreino
pastasMaximo = 10

kMaximo = 50
xTreinoNum = dTreinoFinal[["age","total.people","education.months","capital.gain", "capital.loss", "hours.per.week"]]



print('Tamanho:', xTreinoNum.shape)
#Para guardar os valores de Knn e respectivos scores deste cenário

ks1 = []

scores1 = []

kMelhor1 = 1

melhorScore1 = 0

start = time.time()



#Loop para testar vários valores de Knn na validação cruzada

for k in range(1, kMaximo+1):

    

    knn = KNeighborsClassifier(n_neighbors=k)

    ks1.append(k)     



    scores = cross_val_score(knn, xTreinoNum, yTreino, cv = pastasMaximo)

    mediaScores = np.mean(scores)       #Média dos valores

    scores1.append(mediaScores)





    print('K-nn=', k, '\tScore= ', mediaScores)



    if mediaScores > melhorScore1:

        melhorScore1 = mediaScores

        kMelhor1 = k



stop = time.time()

        

print("\nMelhor valor de K = ", kMelhor1)

print("Melhor Score = ", melhorScore1)  

print("Tempo processamento (ms)=", stop-start)
plt.plot(ks1, scores1,linewidth=3)

plt.title('Valores de K x Scores')

plt.xlabel('K')

plt.ylabel('Scores')
xTreinoNaoNum = dTreinoFinal.copy()

xTreinoNaoNum.drop(["age",

                    "total.people",

                    "education.months",

                    "capital.gain", 

                    "capital.loss", 

                    "hours.per.week",

                    "income"], axis=1, inplace=True)
#Para guardar os valores de Knn e respectivos scores deste cenário

ks2 = []

scores2 = []

kMelhor2 = 1

melhorScore2 = 0

start = time.time()



#Loop para testar vários valores de Knn e Pastas na validação cruzada

for k in range(1, kMaximo+1):

    start_loop = time.time()

    

    knn = KNeighborsClassifier(n_neighbors=k)

    ks2.append(k)     



    scores = cross_val_score(knn, xTreinoNaoNum, yTreino, cv = pastasMaximo)

    mediaScores = np.mean(scores)       #Média dos valores

    scores2.append(mediaScores)



    

    #Condicional para gravar o melhor Score

    if mediaScores > melhorScore2:

        melhorScore2 = mediaScores

        kMelhor2 = k

    

    #Impressão dos resultados do loop

    stop_loop = time.time()

    print('K-nn=', k, '\tScore= ', mediaScores, '\tTempo (ms)=', stop_loop-start_loop)

        

stop = time.time()

        

print("\nMelhor valor de K = ", kMelhor2)

print("Melhor Score = ", melhorScore2)  

print("Tempo processamento (ms)=", stop-start)
plt.plot(ks2, scores2,linewidth=3)

plt.title('Valores de K x Scores')

plt.xlabel('K')

plt.ylabel('Scores')
xTreinoDesigual = dTreinoFinal[['sex_Female', 

                                'sex_Male',

                                'race_Amer-Indian-Eskimo',

                                'race_Asian-Pac-Islander',

                                'race_Black',

                                'race_Other',

                                'race_White']]
#Para guardar os valores de Knn e respectivos scores deste cenário

ks3 = []

scores3 = []

kMelhor3 = 1

melhorScore3 = 0

start = time.time()



#Loop para testar vários valores de Knn e Pastas na validação cruzada

for k in range(1, kMaximo+1):

    start_loop = time.time()



    knn = KNeighborsClassifier(n_neighbors=k)

    ks3.append(k)     



    scores = cross_val_score(knn, xTreinoDesigual, yTreino, cv = pastasMaximo)

    mediaScores = np.mean(scores)       #Média dos valores

    scores3.append(mediaScores)

    

    

    #Condicional para gravar o melhor Score

    if mediaScores > melhorScore3:

        melhorScore3 = mediaScores

        kMelhor3 = k

    

    #Impressão dos resultados do loop

    stop_loop = time.time()

    print('K-nn=', k, '\tScore= ', mediaScores, '\tTempo (ms)=', stop_loop-start_loop)

    



stop = time.time()



print("\nMelhor valor de K= ", kMelhor3)

print("Melhor Score= ", melhorScore3)  

print("Tempo processamento (ms)=", stop-start)
plt.plot(ks3, scores3,linewidth=3)

plt.title('Valores de K x Scores')

plt.xlabel('K')

plt.ylabel('Scores')
xTreinoTempo= dTreinoFinal[['age', 'hours.per.week']]
#Para guardar os valores de Knn e respectivos scores deste cenário

ks4 = []

scores4 = []

kMelhor4 = 1

melhorScore4 = 0

start = time.time()



#Loop para testar vários valores de Knn e Pastas na validação cruzada

for k in range(1, kMaximo+1):

    start_loop = time.time()



    knn = KNeighborsClassifier(n_neighbors=k)

    ks4.append(k)     



    scores = cross_val_score(knn, xTreinoTempo, yTreino, cv = pastasMaximo)

    mediaScores = np.mean(scores)       #Média dos valores

    scores4.append(mediaScores)

    

    

    #Condicional para gravar o melhor Score

    if mediaScores > melhorScore4:

        melhorScore4 = mediaScores

        kMelhor4 = k

    

    #Impressão dos resultados do loop

    stop_loop = time.time()

    print('K-nn=', k, '\tScore= ', mediaScores, '\tTempo (ms)=', stop_loop-start_loop)

    



stop = time.time()



print("\nMelhor valor de K= ", kMelhor4)

print("Melhor Score= ", melhorScore4)  

print("Tempo processamento (ms)=", stop-start)
plt.plot(ks4, scores4,linewidth=3)

plt.title('Valores de K x Scores')

plt.xlabel('K')

plt.ylabel('Scores')
xTreinoTotal = dTreinoFinal.copy()

xTreinoTotal.drop(labels=['income'], axis=1, inplace = True)

xTreinoTotal
#Para guardar os valores de Knn e respectivos scores deste cenário

ks5 = []

scores5 = []

kMelhor5 = 1

melhorScore5 = 0

start = time.time()



#Loop para testar vários valores de Knn e Pastas na validação cruzada

for k in range(1, kMaximo+1):

    start_loop = time.time()



    knn = KNeighborsClassifier(n_neighbors=k)

    ks5.append(k)     



    scores = cross_val_score(knn, xTreinoTotal, yTreino, cv = pastasMaximo)

    mediaScores = np.mean(scores)       #Média dos valores

    scores5.append(mediaScores)



    #Condicional para gravar o melhor Score

    if mediaScores > melhorScore5:

        melhorScore5 = mediaScores

        kMelhor5 = k

    

    #Impressão dos resultados do loop

    stop_loop = time.time()

    print('K-nn=', k, '\tScore= ', mediaScores, '\tTempo (ms)=', stop_loop-start_loop)

    

    

stop = time.time()



print("\nMelhor valor de K = ", kMelhor5)

print("Melhor Score = ", melhorScore5)  

print("Tempo processamento=", stop-start, 'ms')
plt.plot(ks5, scores5,linewidth=3)

plt.title('Valores de K x Scores')

plt.xlabel('K')

plt.ylabel('Scores')
KnnFinal = 32



xTreinoFinal = dTreinoFinal.copy()

xTreinoFinal.drop(labels=['income'], axis=1, inplace = True)



yTreinoFinal = dTreinoFinal['income']
knn = KNeighborsClassifier(n_neighbors=KnnFinal)



#Treinar Modelo

knn.fit(xTreinoFinal, yTreinoFinal)
yTestePredizer = knn.predict(dTesteFinal)



#Visualização das predições

yTestePredizer
savepath = "predictions.csv"

prev = pd.DataFrame(yTestePredizer, columns = ["income"])

prev.to_csv(savepath, index_label="Id")

prev
mediaGlobal = np.average(a=[melhorScore1, melhorScore2,melhorScore3,melhorScore4,melhorScore5])

print(mediaGlobal)