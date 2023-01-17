#Nº USP: 10774354

#Hash: 50
import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import numpy as np
#vamos guradar os endereços das bases de teste e de treino

file_treino="../input/adult-pmr3508/train_data.csv"

file_test="../input/adult-pmr3508/test_data.csv"
#criar as matrizes da base treino e teste

base_treino=pd.read_csv(file_treino,

        names=

        ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

base_teste=pd.read_csv(file_test,

        names=

        ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
#olhar o começo de cada tabela

base_treino.head()
base_teste.head()
#retirar a primeira linha, com os titulos antigos

base_treino.drop(base_treino.index[0],inplace=True)

base_teste.drop(base_teste.index[0],inplace=True)
#verificar o head novamente

base_treino.head()
base_teste.head()
#verificando tamnaho de cada base

base_treino.shape
base_teste.shape
#agora, vamos verificar, visualmente, a distribuição de algumas de nossas variáveis
#distribuição das idades

base_treino["Age"].value_counts().plot(kind='bar')
#distribuição do tipo de trabalho

base_treino["Workclass"].value_counts().plot(kind='pie')
#distribuição do grau de escolaridade

base_treino["Education"].value_counts().plot(kind='pie')
base_treino["Education-Num"].value_counts().plot(kind='bar')
#distribuição de etnias

base_treino["Race"].value_counts().plot(kind='bar')
#distribuição de gênero

base_treino["Sex"].value_counts().plot(kind='bar')
#distribuição de nacionalidade

base_treino["Country"].value_counts()
#agora vamos eliminar as linhas que contenham dados faltantes

N_treino=base_treino.dropna()
#verificar o shape

N_treino.shape
# utilizaremos os dados numéricos para a implementação dos 3 algorítimos

Xtreino=N_treino[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

Ytreino=N_treino[["Target"]]

Xteste =base_teste[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
Ytreino.head()
Xtreino.head()
Xteste.head()
# bibliotecas a serem usadas

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier 
# fazendo a regressão logistica

logreg = LogisticRegression()

logreg.fit(Xtreino,Ytreino)

Ypred=logreg.predict(Xteste)

# avaliação por validacao cruzada

scores = cross_val_score(logreg, Xtreino, Ytreino, cv=10)

med_logreg=np.mean(scores) #media dos valores da validação cruzada

# avaliação da acurácia

Ypredt=logreg.predict(Xtreino)

ac_logreg=accuracy_score(Ytreino,Ypredt,normalize=True,sample_weight=None)

ac_logreg
med_logreg
# fazendo uma floresta aleatória

rand=RandomForestClassifier(n_estimators=100) #utilizando uma floresta de 100 árvores

rand.fit(Xtreino,Ytreino)

Ypred=rand.predict(Xteste)

# avaliação por validacao cruzada

scores = cross_val_score(rand, Xtreino, Ytreino, cv=10)

med_randflor=np.mean(scores) #media dos valores da validação cruzada

# avaliação da acurácia

Ypredt=rand.predict(Xtreino)

ac_randflor=accuracy_score(Ytreino,Ypredt,normalize=True,sample_weight=None)
ac_randflor
med_randflor
# fazendo uma árvore aleatória

arvore = DecisionTreeClassifier()

arvore.fit(Xtreino,Ytreino)

Ypred=arvore.predict(Xteste)

# avaliação por validacao cruzada

scores = cross_val_score(arvore, Xtreino, Ytreino, cv=10)

med_arvore=np.mean(scores) #media dos valores da validação cruzada

# avaliação da acurácia

Ypredt=arvore.predict(Xtreino)

ac_arvore=accuracy_score(Ytreino,Ypredt,normalize=True,sample_weight=None)
ac_arvore
med_arvore
# Com os resultados das 3 classificações acima, vemos que pra este problema,a regressão logística apresentou resultados de validação cruzada e de acurácia consideravelmente inferiores aos outros classificadores. Já a árvore aleatória e a floresta aleatória de 100 árvores apresentaram resultados similares para ambas as métricas (o que não seria o esperado para um problema mais complexo). Considerando que o tempo de processamento de uma árvore aleatória é bem menor que o tempo de uma floresta, dentre os 3 classificadores aqui estudados, o mais indicado para este problema é a árvore aleatória.
# ATIVIDADE EXTRA
#criar as matrizes da base treino e teste

base_treino = pd.read_csv('../input/at-extra/train(extra).csv', na_values='?').reset_index(drop = True)

base_teste = pd.read_csv('../input/at-extra/test(extra).csv', na_values='?').reset_index(drop = True)

#olhar o começo de cada tabela

base_treino.head()
base_teste.head()
#verificando tamnaho de cada base

base_treino.shape
base_teste.shape
#agora, vamos verificar, visualmente, a distribuição de algumas de nossas variáveis
#distribuição espacial dos dados (longitude e latitude)

x=base_treino['longitude']

y=base_treino['latitude']

plt.scatter(x, y)

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.show()
x=base_teste['longitude']

y=base_teste['latitude']

plt.scatter(x, y)

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.show()
#agora vamos ver a distribuição espacial de cada variável na base de treino

x=base_treino['longitude']

y=base_treino['latitude']

size=base_treino['median_age']

plt.scatter(x, y,s=15*size/max(base_treino['median_age']))

plt.title('Distribuição etária')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.show()
size=base_treino['total_rooms']

plt.scatter(x, y,s=15*size/max(base_treino['total_rooms']))

plt.title('Total de salas')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.show()
size=base_treino['total_bedrooms']

plt.scatter(x, y,s=15*size/max(base_treino['total_bedrooms']))

plt.title('Total de quartos')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.show()
size=base_treino['population']

plt.scatter(x, y,s=15*size/max(base_treino['population']))

plt.title('Distribuição populacional')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.show()
size=base_treino['median_income']

plt.scatter(x, y,s=15*size/max(base_treino['median_income']))

plt.title('Renda média')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.show()
size=base_treino['median_house_value']

plt.scatter(x, y,s=15*size/max(base_treino['median_house_value']))

plt.title('Valor médio das casas')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.show()
#bibliotecas a serem usadas para as regressoes

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
#iremos utilizar 3 regressores: regressão linear, polinomial com grau baixo (<=5) e polinomial com grau alto (>=10)
#o código a seguir era a minha ideia inicial, porém não consegui o rodar. Contudo o deixarei mesmo assim abaixo, como forma de comentário: 



##vamos separar as features para posterior seleção de variáveis

#v=[base_treino["longitude"],base_treino["latitude"],base_treino["median_age"],base_treino["total_rooms"],base_treino["total_bedrooms"],base_treino["population"],base_treino["households"],base_treino["median_income"]]

##a seleção de variáveis seguirá uma logíca de foward selection, a partir da regressão linear. As features determinadas para a regressão linear, se manterão para as demais regressões

##Regressão linear + seleção de features

#lin=LinearRegression()

#X_treino=[]

#X_treino_aux=[]

#X_teste=[]

#Y_treino=base_treino[["median_house_value"]]

#r2=0

#r2_aux=0

#i=0

#for k=1 to 5:  #iremos selecionar 5 atributos  

#    if i <8:

#        X_treino.append(v[i])

#        lin.fit(X_treino,Y_treino)

#        r2=lin.score(X_treino,Y_treino)

#        if r2>r2_aux:

#            r2_aux=r2

#            X_treino_aux=X_treino
#utilizaremos os atributos idade, população, renda média a posição espacial (longitude e latitude) para as regressões

X_treino=base_treino[["longitude","latitude","median_age","population","median_income"]]

Y_treino=base_treino[["median_house_value"]]

X_teste=base_teste[["longitude","latitude","median_age","population","median_income"]]

#Regressão linear

lin=LinearRegression()

lin.fit(X_treino,Y_treino)

Y_teste=lin.predict(X_teste)

#coeficiente de determinação (r^2)

r2_lin=lin.score(X_treino,Y_treino)
Y_teste
r2_lin
#Regressão Polinomial de grau baixo

#para fazer uma regressão polinomial, nós precisamos modificar nossa bas de features (x) de tal modo que as potências de cada valor também sejam expressadas (x^n)

for i in range (2,6): #iremos testar polinomios de grau 2 a 5

    trans = PolynomialFeatures(degree=i, include_bias=False) #essas variável vai adicionar aos dados x, colunas com os dados x^2, x^3,..., x^î

    trans.fit(X_treino)

    X_treino_trans=trans.transform(X_treino) #agora já temos nossas features transformada. Faremos então uma regressão linear com essas "novas" features

    pol=LinearRegression().fit(X_treino_trans, Y_treino)

    r2_pol=pol.score(X_treino_trans, Y_treino) #guardando o resultado do r^2 (coeficiente de determinação) 

    X_teste_trans=trans.transform(X_teste) #tranformando a base teste para prever o resultado (Y_teste)

    Y_teste=pol.predict(X_teste_trans)

    print("Resultados para grau %d:\n"%i)

    print("Previsão:\n")

    print(Y_teste)

    print("Coeficiene de determinação:\n")

    print(r2_pol)
#Regressão Polinomial de grau alto

#para fazer uma regressão polinomial, nós precisamos modificar nossa bas de features (x) de tal modo que as potências de cada valor também sejam expressadas (x^n)

for i in range (11,14): #iremos testar polinomios de grau 11 a 13

    trans = PolynomialFeatures(degree=i, include_bias=False) #essas variável vai adicionar aos dados x, colunas com os dados x^2, x^3,..., x^î

    trans.fit(X_treino)

    X_treino_trans=trans.transform(X_treino) #agora já temos nossas features transformada. Faremos então uma regressão linear com essas "novas" features

    pol=LinearRegression().fit(X_treino_trans, Y_treino)

    r2_pol=pol.score(X_treino_trans, Y_treino) #guardando o resultado do r^2 (coeficiente de determinação) 

    X_teste_trans=trans.transform(X_teste) #tranformando a base teste para prever o resultado (Y_teste)

    Y_teste=pol.predict(X_teste_trans)

    print("Resultados para grau %d:\n"%i)

    print("Previsão:\n")

    print(Y_teste)

    print("Coeficiene de determinação:\n")

    print(r2_pol)
#o coeficiente de determinação negativo para graus 5 e 11 é explicado pela fonte https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html como sendo representativo de um modelo arbitrariamente ruim
#Com os resultados de coeficiente de determinação dos 3 regressores testados, percebemos que este modelo funciona melhor para polinômios de grau menor que 5. Para tais, o seu valor de r^2 se mostra crescente, atingindo seu máximo em um polinômio de grau 4 (~0,644). A partir daí, o modelo sofre uma brusca queda de desempenho (r^2<0,3)