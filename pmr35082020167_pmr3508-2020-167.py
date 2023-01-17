import numpy as np # linear algebra

import pandas as pd # processamento de dados e handling de CSV

import sklearn # k-nn methods and other machine learning utilities

import matplotlib.pyplot as plt #plotar gráficos e visualizar dados

import seaborn as sns #visualizar dados





#importando base de teste

teste = pd.read_csv("../input/adult-pmr3508/test_data.csv",

        index_col=['Id'],

        na_values="?")



#importando base de treino

treino = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        index_col=['Id'],

        na_values="?")
#verificar o tamanho da tabela (linhas e colunas)

treino.shape



#visualizando rapidamente a tabela em si

treino.head()



#vendo os tipos de dados da tabela

treino.info()

teste.info()

#pequena manipulação para construção do gráfico

substitution = {"<=50K": 0, ">50K": 1}

treino.income = [substitution[i] for i in treino.income]



plt.figure(figsize=(10,10))

sns.heatmap(treino.corr(method="pearson"), square=True, annot=True, cbar=True, vmin=-1, vmax=1, cmap="Greens")

plt.show()

#analisando sexo

sns.catplot(y="sex", x="income", kind="bar", data=treino)
#analisando etnia

sns.catplot(y="race", x="income", kind="bar", data=treino)
#analisando país nativo

sns.catplot(y="native.country", x="income", kind="bar", data=treino)
#analisando estado civil

sns.catplot(y="marital.status", x="income", kind="bar", data=treino)

#analisando relacionamento

sns.catplot(y="relationship", x="income", kind="bar", data=treino)

#analisando grau de escolaridade

sns.catplot(y="education", x="income", kind="bar", data=treino)



#analisando ocupação

sns.catplot(y="occupation", x="income", kind="bar", data=treino)

#analisando workclass

sns.catplot(y="workclass", x="income", kind="bar", data=treino)

#removendo dados faltantes da base adult

nTreino = treino.dropna()

nTreino.shape

#transformando dados faltantes em string para evitar erros de processamento posteriormente



for i in range(0,16280):

    if pd.isnull(teste["workclass"][i]) or pd.isnull(teste["occupation"][i]):

        teste["workclass"][i] = str(teste["workclass"][i])

        teste["occupation"][i] = str(teste["occupation"][i])



        

nTreino = nTreino.drop(['fnlwgt', 'native.country', 'education'], axis=1)

nTeste = teste.drop(['fnlwgt', 'native.country', 'education'], axis=1)

from sklearn import preprocessing



numTreino = nTreino

numTeste = nTeste



numTreino = numTreino.apply(preprocessing.LabelEncoder().fit_transform)

numTeste = numTeste.apply(preprocessing.LabelEncoder().fit_transform)



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score





Ytreino = numTreino.pop('income')

Xtreino = numTreino

Xteste = numTeste



maxK = 40

bestResult = [0,0]





for testNumber in range(10,maxK):

        knn = KNeighborsClassifier(n_neighbors=testNumber)

        # fazendo a validação cruzada

        accuracy = cross_val_score(knn, Xtreino, Ytreino, cv=10,scoring="accuracy").mean()

        print("Para K =",testNumber,", acurácia de",accuracy)

        # guardando o nosso melhor resultado numa lista para conseguirmos verificar depois

        if accuracy > bestResult[1]:

                bestResult = [testNumber, accuracy]



print(bestResult)

maxKnn = KNeighborsClassifier(n_neighbors=bestResult[0])

maxKnn.fit(Xtreino, Ytreino)



numPrediction = maxKnn.predict(Xteste)



backSubstitution = {0: '<=50K', 1: '>50K'}

prediction = np.array([backSubstitution[i] for i in numPrediction], dtype=object)

print(prediction)

submission = pd.DataFrame()

submission[0] = nTeste.index

submission[1] = prediction

submission.columns = ['Id', 'Income']



submission.to_csv('submission.csv', index = False)
