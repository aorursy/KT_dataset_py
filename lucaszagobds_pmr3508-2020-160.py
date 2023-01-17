#Saudações! O objetivo deste notebook é analisar e discutir os dados contidos na base adult

#bem como a utilização destes dados para testar a acuracia do algoritmo KNN como classificador.

#O primeiro passo é a importação das bibliotecas a serem utilizadas:

import pandas as pd       #biblioteca com diversas funções úteis 

import sklearn            #biblioteca que será utilizada para a estimação do classificador KNN

import matplotlib.pyplot as plt      #biblioteca para criação de gráficos e visualização de dados

import numpy as np        #biblioteca que permite acesso a várias funções matemáticas
#Os dados da base adult precisarão ser devidamente importados antes de serem exibidos

#Dados de treino e dados de teste não devem ser misturados

dados_treino=pd.read_csv("../input/adult-pmr3508/train_data.csv",

                        names=[

"ID","Idade","Tipo_Profissional","fnlwgt","Educação","n_educação","Estado_Civil","Ocupação","Relacionamento",

    "Raça", "Sexo", "Renda_Ganha", "Renda_Perdida","Horas_por_Semana", "País", "Alvo"],     #Rótulos

    sep=r'\s*,\s*',

    engine='python',  

    na_values="?")          #Dados faltantes recebem '?'



dados_teste=pd.read_csv("../input/adult-pmr3508/test_data.csv", 

                        names=[

"ID","Idade","Tipo_Profissional","fnlwgt","Educação","n_educação","Estado_Civil","Ocupação","Relacionamento",

    "Raça", "Sexo", "Renda_Ganha", "Renda_Perdida","Horas_por_Semana", "País","Alvo"],     #Rótulos

    sep=r'\s*,\s*',

    engine='python',  

    na_values="?")          #Dados faltantes recebem '?'
#Dessa forma, temos:

dados_treino.head()
dados_teste.head()
#Eliminando a linha redundante: 

dados_treino.drop(dados_treino.index[0],inplace=True)

dados_teste.drop(dados_teste.index[0],inplace=True)

dados_treino.head()
dados_teste.head()
#Agora que os dados estão devidamente apresentados, pode-se começar a formular análises a respeito deles.

#Esta  próxima etapa será feita de análises a respeito dos dados processados anteriormente:







dados_treino["Ocupação"].value_counts().plot(kind="bar")
#No gráfico acima pode-se constatar que, dentre os dados pessoais colhetados,

#a minoria das pessoas tem uma ocupação das forças armadas, acompanhado dos serviços domésticos privados.





cruzar=pd.crosstab(dados_treino['Idade'], dados_treino['Alvo'])

cruzar.plot()
#Para entender o gráfico acima é necessário entender o rótulo "Alvo". 

#Este rótulo indica se a pessoa cujo os dados estão vinculados ganham mais do que 50K por ano ou não.

#No gráfico em questão, pode-se afirmar que a maior parte das pessoas que ganham essa quantia têm em torno de 30 a 40 anos.





cruzar2 = pd.crosstab(dados_treino['Sexo'], dados_treino['Alvo'])

cruzar2.plot(kind="bar")
#O gráfico acima indica que a maoiria dos homens e das mulheres ganham menos do que 50K.



cruzar3 = pd.crosstab(dados_treino["Educação"],dados_treino["Alvo"])

cruzar3.plot(kind='bar')
#Naturalmente, o gráfico aponta que pessoas que possuem maior grau educacional tendem a receber mais

#Nota-se ainda que uma grande quantidade dos que só possuem ensino médio completo ganham abaixo dos 50K



cruzar4 = pd.crosstab(dados_treino["País"],dados_treino["Alvo"])

cruzar4.plot(kind='bar')
#Uma quantidade altíssima das pessoas são dos Estados Unidos. Percebe-se ainda que a maioria das pessoas ganham menos do que 50K





cruzar5 = pd.crosstab(dados_treino["Raça"], dados_treino["Alvo"])

cruzar5.plot(kind='bar')
#Percebe-se que pessoas da raça branca desses dados selecionados são a maioria 

#dentre elas, a maior parte ganha menos do que 50K



cruzar6 = pd.crosstab(dados_treino["Relacionamento"],dados_treino["Alvo"])

cruzar6.plot(kind='bar')
#Interessantemente, a maior parte dos que recebem mais do que 50K são homens casados(maridos)



cruzar7 = pd.crosstab(dados_treino["Estado_Civil"],dados_treino["Alvo"])

cruzar7.plot(kind='bar')
#Outro gráfico interessante: Pessoas que nunca se casaram tendem a ganhar menos do que pessoas classificadas como 

# "Cônjuge civil"

#Para a preparação dos testes para o KNN será adequado que se remova os dados faltantes dos dados de teste

#Primeiro a verificação de onde eles estão:

dados_teste.isnull().mean(axis=0)
#Serão retirados através do processo de amputação por moda

dados_teste2=dados_teste

rotulos = ["Tipo_Profissional","Ocupação", "País"]

for i in rotulos:

    moda = dados_teste2[i].describe().top

    dados_teste2[i] = dados_teste2[i].fillna(moda)
dados_teste2.isnull().mean(axis=0)  #conferindo
#É possível transformar rótulos com categorias em forma de texto para números:



gênero = ["Male","Female"] 



n = ["3","1"]    #Esses números atribuídos são a porcentagem de uma categoria em comparação a outra

raça = ["Asian-Pac-Islander","White","Black","Amer-Indian-Eskimo","Other"]

p=["26", "25", "12", "11","9"]

gênero += raça

n += p 

#Atribuição de um número para cada categoria do rótulo em questão

def função(rótulo):

    for i in range(len(n)):

        if rótulo == gênero[i]:

            return n[i]

    return rótulo



dados_treino["Sexo"]=dados_treino["Sexo"].apply(função)

dados_treino["Raça"]=dados_treino["Raça"].apply(função)

dados_teste2["Sexo"]=dados_teste2["Sexo"].apply(função)

dados_teste2["Raça"]=dados_teste2["Raça"].apply(função)

#Para os dados faltantes dos dados de treino pode-se utilizar a seguinte função:

dados_treino2=dados_treino.dropna()
#Note que, de fato, há diferença entre os valores:

dados_treino["Ocupação"].value_counts()
dados_treino2["Ocupação"].value_counts()
#Agora começam os testes com o classificador KNN















#Para este primeiro teste serão utilizados os rótulos "Idade","Horas_por_Semana","n_educação", "fnlwgt" e "Renda_Perdida"

Xdados=dados_treino2[["Idade","Horas_por_Semana","n_educação","fnlwgt","Renda_Perdida"]]

Ydados=dados_treino2.Alvo

Xteste=dados_teste2[["Idade","Horas_por_Semana","n_educação","fnlwgt","Renda_Perdida"]]
from sklearn.neighbors import KNeighborsClassifier
#Selecionando-se um valor arbitrário para K

knn = KNeighborsClassifier(n_neighbors=10)
#E se valendo da validação cruzada

from sklearn.model_selection import cross_val_score
#após alguns processos para o funcionamento do classificador teremos uma acurácia





resultados = cross_val_score(knn, Xdados, Ydados, cv=10)
resultados
knn.fit(Xdados,Ydados)
testeY=knn.predict(Xteste)
testeY
from sklearn.metrics import accuracy_score
acurácia=np.mean(resultados)

acurácia
#O classificador obteve 0,74 de acuaracia aproximadamente com os dados de teste. 

#Variando-se os valores de K e de cv talvez seja possível chegar em um resultado melhor 

knn = KNeighborsClassifier(n_neighbors=30)

resultados = cross_val_score(knn, Xdados, Ydados, cv=11)

resultados
knn.fit(Xdados,Ydados)
testeY=knn.predict(Xteste)

testeY
acurácia=np.mean(resultados)

acurácia
#Uma acurácia melhor foi obtida. É possível alterar os rótulos para obter novas acurácias

#Um novo teste será realizado para comprovar isto

Xdados=dados_treino2[["Idade","n_educação","fnlwgt","Sexo","Renda_Ganha","Renda_Perdida","Horas_por_Semana"]]

Ydados=dados_treino2.Alvo

Xteste=dados_teste2[["Idade","n_educação","fnlwgt","Sexo","Renda_Ganha","Renda_Perdida","Horas_por_Semana"]]

knn = KNeighborsClassifier(n_neighbors=23,p=1)

resultados = cross_val_score(knn, Xdados, Ydados, cv=16)

resultados

knn.fit(Xdados,Ydados)

testeY=knn.predict(Xteste)

testeY
acurácia=np.mean(resultados)

acurácia
#Uma acurácia de 79% foi obtida. O progresso agora será submetido:

id_index = pd.DataFrame({'Id' : list(range(len(testeY)))})

income = pd.DataFrame({'income' : testeY})

result = income

result
result.to_csv("submission.csv", index = True, index_label = 'Id')