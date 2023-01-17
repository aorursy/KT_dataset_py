# Importando as bibliotecas padrão 



# Manipulação de dados

import pandas as pd

import numpy as np



# Visualização de dados

import matplotlib.pyplot as plt

import seaborn as sns



# Machine Learning 

from sklearn import metrics # analisa a acurácia de nossos modelos



# Ocultando Warnings indesejados

import warnings

warnings.filterwarnings('ignore')
# importando nossa base de dados

from sklearn.datasets import load_breast_cancer #importando a base de dados nativas no sklearn



dados=load_breast_cancer() # Carregando base de dados



# vamos ver a descrição de nossa base de dados

print(dados.DESCR)
# Tranformando a base de dados em um DataFrame



cancer=pd.DataFrame(data=dados.data, columns=dados.feature_names) # convertendo para dataframe com ajuda do Pandas



cancer['Class']=dados.target # Adicionando a nossa Target
# Um dataframe Pandas parece muito com uma tabela Excel



cancer.head(3) # Visualizando as 3 primeiras linhas de nosso dataframe
# Vamos começar descobrindo as dimensões de nosso dataframe - Linhas X Colunas

cancer.shape # nosso df tem 569 linhas distribuidas entre 31 colunas
# Distribuição de nossas classes

cancer['Class'].value_counts() # 1- Benigno 0 = Máligno
# Temos a informação: 357 casos benignos e 212 casos malignos.



# Vamos visualizar isso melhor

# Criando um Gráfico de Pizza - ou no inglês um gráfico de torta(PiePlot)



colors=['#35b2de','#ffcb5a'] # Apenas escolhando as cores



labels=cancer['Class'].value_counts().index

plt.pie(cancer['Class'].value_counts(),autopct='%1.1f%%',colors=colors) # conta as ocorrências de cada classe e exibe a porcentagem

plt.legend(labels,bbox_to_anchor=(1.25,1),) # Nossas Legendas

plt.title('Porcentagem: Benignos x Malignos ')

plt.show()
# Misssing Values

cancer.isnull().sum() 
# Vamos criar nossas amostras para a construção dos modelos

# Vamos usar mais uma vez a biblioteca sklearn

from sklearn.model_selection import train_test_split



# primeiro vamos dividir nossa base de dados entre features e target

X= cancer.iloc[:,0:-1]# Selecionando todas as linhas, da primeira coluna até a penúltima coluna.

Y=cancer.iloc[:,-1] # Selecionando todas as linhas da última coluna ['Class'].





x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=42)

# test-size: neste casos vamos dividir nosso dataset em 70% treino e 30% teste

# random_state: vamos selecionar de forma aleatória
# Agora temos nossas bases de dados para treino e testes 

print('X treino',x_train.shape)

print('X test',x_test.shape)

print('Y treino',y_train.shape)

print('Y test',y_test.shape)
# importando nosso modelo

from sklearn.linear_model import LogisticRegression



logreg=LogisticRegression() # Criando o modelo

logreg.fit(x_train,y_train) # Treinando o modelo

y_pred=logreg.predict(x_test) # predizendo

acc_logreg=round(metrics.accuracy_score(y_pred,y_test)*100,1) # avaliando a acurácia. previsões x resultados reais

print("{}% de acurácia".format(acc_logreg,))
# importando nosso modelo

from sklearn.svm import SVC



svc=SVC() # Criando o modelo

svc.fit(x_train,y_train) # Treinando o modelo

y_pred=svc.predict(x_test) # predizendo

acc_svc=round(metrics.accuracy_score(y_pred,y_test)*100,1) # avaliando a acurácia. previsões x resultados reais

print(acc_svc,"% de acurácia")
# importando nosso modelo

from sklearn.naive_bayes import GaussianNB



gaussian=GaussianNB() # Criando o modelo

gaussian.fit(x_train,y_train) # Treinando o modelo

y_pred=gaussian.predict(x_test) # predizendo

acc_gaussian=round(metrics.accuracy_score(y_pred,y_test)*100,1) # avaliando a acurácia. previsões x resultados reais

print(acc_gaussian,"% de acurácia")
# importando nosso modelo

from sklearn.tree import DecisionTreeClassifier



tree=DecisionTreeClassifier() # Criando o modelo

tree.fit(x_train,y_train) # Treinando o modelo

y_pred=tree.predict(x_test) # predizendo

acc_tree=round(metrics.accuracy_score(y_pred,y_test)*100,1) # avaliando a acurácia. previsões x resultados reais

print(acc_tree,"% de acurácia")
# importando nosso modelo

from sklearn.ensemble import RandomForestClassifier



forest=RandomForestClassifier(n_estimators=100) # Criando o modelo

forest.fit(x_train,y_train) # Treinando o modelo

y_pred=forest.predict(x_test) # predizendo

acc_forest=round(metrics.accuracy_score(y_pred,y_test)*100,1) # avaliando a acurácia. previsões x resultados reais

print(acc_forest,"% de acurácia")
# importando nosso modelo

from sklearn.neighbors import KNeighborsClassifier



knn=KNeighborsClassifier(n_neighbors=1) # Criando o nosso classificador

knn.fit(x_train,y_train) # Treinando o modelo

y_pred=knn.predict(x_test) # Predizendo nossos dados de testet

acc_knn=round(metrics.accuracy_score(y_pred,y_test)*100,1)

print(acc_knn,"% de acurácia") # Exibindo resultado
# É Simples, vamos construir modelos KNN dentro de um for, e testar qual são os melhores resultados.



k_range=range(1,25) # vamos testar n_neighbors de 1 a 25

scores=[] # vamos armazenar os resultados aqui



for k in k_range:

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train,y_train)

    y_pred=knn.predict(x_test)

    scores.append(metrics.accuracy_score(y_pred,y_test))



# Por último vamos gerar uma visualização para chegarmor ao veredito.

plt.plot(k_range,scores)

plt.xlabel("Valor de K para KNN")

plt.ylabel("Teste de Acurácia")

modelos=pd.DataFrame({'Modelos':['Regressão Logística','Support Vector Machine',\

                    'Gaussian Naive Bayes','Árvore de Decisão',\

                    'Random Forest','KNN'],\

         'Score':[acc_logreg,acc_svc,acc_gaussian,acc_tree,acc_forest,acc_knn]})



modelos.sort_values(by="Score", ascending=False)