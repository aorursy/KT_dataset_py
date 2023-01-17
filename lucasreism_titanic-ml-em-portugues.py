#Importando Bibliotecas

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix

import itertools

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



#Importando os dados para as variáveis= pd.readcsv("../input/train.csv")

treino =  pd.read_csv('../input/repository/lucasreism-titanic-em-portugues-bbe38eb/train.csv')

teste =  pd.read_csv('../input/repository/lucasreism-titanic-em-portugues-bbe38eb/test.csv')



#Colunas Renomeadas

treino = treino.rename(columns={'passengerid': 'Passengerid', 'survived': 'Survived', 'pclass': 'Pclass', 'name':'Name', 'sex': 'Sex', 'age': 'Age', 'sibsp': 'Sibsp', 'parch':'Parch','ticket': 'Ticket', 'fare':'Fare', 'cabin': 'Cabin', 'embarked': 'Embarked'})



#Lista que contém nossas matrizes

todosdados = [treino,teste]
#Colunas e informações sobre o arquivo Treino

print(treino.columns)

print('_'*60)

print(treino.head(5))

print('_'*60)

print(treino.tail(5))

print('_'*60)

print(treino.shape)
#Colunas e informações sobre o arquivo Teste

print(teste.columns)

print('_'*60)

print(teste.head(5))

print('_'*60)

print(teste.tail(5))

print('_'*60)

print(teste.shape)
#Colunas e informações sobre o arquivo Teste

print(treino.columns.values)

print('_'*60)

treino.info()

print('_'*60)

teste.info()
#Dados estatísticos para análise de colunas numéricas

treino.describe()
#Dados estatísticos para análise de colunas com texto

treino.describe(include=['O'])
#Quantidade de dados faltando em cada coluna do arquivo treino

treino.isnull().sum()
#treino.plot(kind='scatter', x='Age', y='Passengerid', rot=70)

#Removendo outlier por Interquartile range : https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm



def remove_outlier(dataset, nome_col):

    q1 = dataset[nome_col].quantile(0.25)

    q3 = dataset[nome_col].quantile(0.75)

    iqr = q3-q1 #Interquartile range

    qbaixo  = q1-1.5*iqr

    qcima = q3+1.5*iqr

    dataset_saida = dataset.loc[(dataset[nome_col] > qbaixo) & (dataset[nome_col] < qcima)]

    print(qbaixo)

    print(qcima)

    return dataset_saida





treino = remove_outlier(treino, 'Age')

teste  = remove_outlier(teste, 'Age')



#remover outliers da coluna 'Fare'

treino = remove_outlier(treino, 'Fare')

teste  = remove_outlier(teste, 'Fare')





#Foi ultilizada a mediana por razão da distribuição dos dados 'Age'

#Preencher os dados faltando com a mediana da idade

for dataset in todosdados:

    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())  



##treino.['Age'].transform(lambda s: 1 if s  0 else 0 )    

treino.boxplot( column='Age', figsize = (12,8))

#Temos idades que estão com um '0.' na frente

#Assumindo que as idades com decimal foram retiradas pela diferença do tempo de duas datas



#treino['Age'] = treino['Age'].astype(int)

treino.describe()
#Quantidade de dados faltando

treino.isnull().sum()
#Quantidade de dados faltando

teste.isnull().sum()
#Retirando as colunas Ticket

treino = treino.drop(['Ticket'], axis=1)

teste = teste.drop(['Ticket'], axis=1)

treino.head()



#Observado que falta o preço da passagem da pessoa no dataset Teste que embarcou em Southampton com classe econômica

#Pegar a mediana de todos os que embarcaram em Southampton com classe econômica

embarcouS_classe3 = teste['Fare'].loc[(teste['Pclass']==3) & (teste['Embarked']=='S')] + treino['Fare'].loc[(treino['Pclass']==3) & (treino['Embarked']=='S')]



#Preencher o valor com essa mediana

teste['Fare'] = teste['Fare'].fillna(embarcouS_classe3.median())



#Quantidades de dados faltando

teste.isnull().sum()
#Dois dados preenchidos com o local de maior frequência

mais_embarcado = treino['Embarked'].value_counts().index[0]

treino['Embarked'] = treino['Embarked'].fillna(mais_embarcado)

treino['Embarked'].describe()



#Preenchendo a classe que estava faltando com a classe mais comprada

treino['Pclass'] = treino['Pclass'].fillna(treino['Pclass'].median()) 

treino['Pclass'].value_counts()
print(treino.Sex.value_counts())



#Todos os títulos que estão padronizados entre uma virgula e um ponto.

treino['New_Sex'] = treino.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())



#Diccionário Homem/Mulher

relacao_titulo_sex = {

    "Capt":       "male",

    "Col":        "male",

    "Major":      "male",

    "Jonkheer":   "male",

    "Don":        "male",

    "Sir" :       "male",

    "Dr":         "male",

    "Rev":        "male",

    "the Countess":"female",

    "Dona":       "female",

    "Mme":        "female",

    "Mlle":       "female",

    "Ms":         "female",

    "Mr" :        "male",

    "Mrs" :       "female",

    "Miss" :      "female",

    "Master" :    "male",

    "Lady" :      "female"

}



#Mapeamento da nova coluna pelo dicionário criado acima

treino.New_Sex = treino.New_Sex.map(relacao_titulo_sex)

print(treino.New_Sex.value_counts())
treino['Titulo'] = treino.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

relacao_titulo = {

    "Capt":       "Oficial",

    "Col":        "Oficial",

    "Major":      "Oficial",

    "Jonkheer":   "Realeza",

    "Don":        "Realeza",

    "Sir" :       "Realeza",

    "Dr":         "Oficial",

    "Rev":        "Oficial",

    "the Countess":"Realeza",

    "Dona":       "Realeza",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Realeza"

}



treino['Titulo'] = treino.Titulo.map(relacao_titulo)



teste['Titulo'] = teste.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())



teste['Name'] = teste.Titulo.map(relacao_titulo)
treino.Cabin.fillna('0', inplace=True)

treino.loc[treino.Cabin.str[0] == 'A', 'Cabin'] = 1

treino.loc[treino.Cabin.str[0] == 'B', 'Cabin'] = 2

treino.loc[treino.Cabin.str[0] == 'C', 'Cabin'] = 3

treino.loc[treino.Cabin.str[0] == 'D', 'Cabin'] = 4

treino.loc[treino.Cabin.str[0] == 'E', 'Cabin'] = 5

treino.loc[treino.Cabin.str[0] == 'F', 'Cabin'] = 6

treino.loc[treino.Cabin.str[0] == 'G', 'Cabin'] = 7

treino.loc[treino.Cabin.str[0] == 'T', 'Cabin'] = 8



teste.Cabin.fillna('0', inplace=True)

teste.loc[teste.Cabin.str[0] == 'A', 'Cabin'] = 1

teste.loc[teste.Cabin.str[0] == 'B', 'Cabin'] = 2

teste.loc[teste.Cabin.str[0] == 'C', 'Cabin'] = 3

teste.loc[teste.Cabin.str[0] == 'D', 'Cabin'] = 4

teste.loc[teste.Cabin.str[0] == 'E', 'Cabin'] = 5

teste.loc[teste.Cabin.str[0] == 'F', 'Cabin'] = 6

teste.loc[teste.Cabin.str[0] == 'G', 'Cabin'] = 7

teste.loc[teste.Cabin.str[0] == 'T', 'Cabin'] = 8
teste['Family'] = teste['SibSp'] + teste['Parch'] 

teste['Sozinho'] = teste['Family'].map(lambda s: 1 if s == 0 else 0)

teste['Pequena_familia'] = teste['Family'].map(lambda s: 1 if 1 <= s <= 3 else 0)

teste['Grande_familia'] = teste['Family'].map(lambda s: 1 if 4 <= s else 0)



treino['Family'] = treino['Sibsp'] + treino['Parch'] 

treino['Sozinho'] = treino['Family'].map(lambda s: 1 if s == 0 else 0)

treino['Pequena_familia'] = treino['Family'].map(lambda s: 1 if 1 <= s <= 3 else 0)

treino['Grande_familia'] = treino['Family'].map(lambda s: 1 if 4 <= s else 0)

treino.head()
#Quantidades de dados faltando

treino.isnull().sum()
#Quantidades de dados faltando

teste.isnull().sum()
treino.info()

teste.info()
sns.barplot(x="Cabin", y="Survived", data=treino,

            label="Cabine vs Taxa de sobrevivência",palette="Blues_d", ci= None)
sns.barplot(x= treino['Titulo'] , y=treino['Survived'], palette="Blues_d", ci= None)
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,10))



sns.barplot(x = 'New_Sex', y = 'Survived', hue = 'Embarked', palette="Blues_d", data=treino, ax = ax1, ci= None)

ax1.set_title('Sexo vs Embarked comparação de sobrevivência')



sns.barplot(x = 'New_Sex', y = 'Survived', hue = 'Pclass',palette="Blues_d", data=treino, ax  =  ax2, ci= None)

ax2.set_title('Sex vs Pclass comparação de sobrevivência')



sns.barplot(x = 'New_Sex', y = 'Survived', hue = 'Sozinho',palette="Blues_d", data=treino, ax  = ax3, ci= None)

ax3.set_title('Sex vs Sozinho comparação de sobrevivência')
sns.pointplot(x="Family", y="Survived", hue="New_Sex", data=treino, palette={"male": "blue", "female": "c"}, ci=None)
treino.hist(column='Age', figsize = (12,5))
facet = sns.FacetGrid(treino, hue="Survived",aspect=4, height  = 3)

facet.map(sns.kdeplot,'Age',shade= True)

facet.add_legend()
#Transformaremos a coluna sexo para Masculino: 0 e Feminino:1

mapeamento_sexo = {'male': 0, 'female':1}

treino['New_Sex'] = treino['New_Sex'].map(mapeamento_sexo)

teste['Sex'] = teste['Sex'].map(mapeamento_sexo)
#Transformação usando a funcão 'get_dummies' que nos dará uma coluna para cada valor único em cada coluna

treino = pd.get_dummies(treino, columns=['Embarked'])

teste = pd.get_dummies(teste, columns=['Embarked'])

treino = pd.get_dummies(treino, columns=['Pclass'])

teste = pd.get_dummies(teste, columns=['Pclass'])

treino = pd.get_dummies(treino, columns=['Titulo'])

teste = pd.get_dummies(teste, columns=['Titulo'])

#treino = pd.get_dummies(treino, columns=['Tem_Cabine'])

#teste = pd.get_dummies(teste, columns=['Tem_Cabine'])





treino.head()
teste.columns
treino.columns
teste = teste.rename(columns={'Sex': 'New_Sex'})

treino = treino.rename(columns ={'Sibsp': 'SibSp','Pclass_1.0':'Pclass_1','Pclass_2.0':'Pclass_2', 'Pclass_3.0':'Pclass_3'})

treino.head()
#Precisarei transformar meus valores em tipo 'Float' para classificação.

treino = treino.astype({'New_Sex': float, 'Embarked_C': float, 'Embarked_Q': float, 'Embarked_S': float, 'Pclass_1': float,

                        'Pclass_2': float,'Pclass_3': float,'Sozinho': float,'Pequena_familia': float,'Grande_familia': float,

                        'Titulo_Master': float,'Titulo_Miss': float,'Titulo_Mrs': float,'Titulo_Mr': float,

                        'Cabin': float })

teste = teste.astype({'New_Sex': float, 'Embarked_C': float, 'Embarked_Q': float, 'Embarked_S': float, 'Pclass_1': float,

                        'Pclass_2': float,'Pclass_3': float,'Sozinho': float,'Pequena_familia': float,'Grande_familia': float,

                        'Titulo_Master': float,'Titulo_Miss': float,'Titulo_Mrs': float,'Titulo_Mr': float,

                        'Cabin': float })
treino.info()
x_treino=treino[['Age', 'Fare', 'New_Sex', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Pclass_1','Pclass_2','Pclass_3',

                 'Sozinho','Pequena_familia','Grande_familia','Titulo_Master','Titulo_Miss','Titulo_Mrs','Titulo_Mr',

                 'Cabin']]

y_treino=treino['Survived']

x_teste=treino[['Age', 'Fare', 'New_Sex', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Pclass_1','Pclass_2','Pclass_3',

                'Sozinho','Pequena_familia','Grande_familia','Titulo_Master','Titulo_Miss','Titulo_Mrs','Titulo_Mr',

                'Cabin']]



x_treino, X_teste_A, y_treino, Y_teste_A = train_test_split(treino[['Age', 'Fare', 'New_Sex', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Pclass_1','Pclass_2','Pclass_3',

                 'Sozinho','Pequena_familia','Grande_familia','Titulo_Master','Titulo_Miss','Titulo_Mrs','Titulo_Mr',

                 'Cabin']], treino['Survived'], random_state=1, test_size=0.3)

#Algoritmo de Floresta aleatória

rnf_A=RandomForestClassifier()



#Aprendizado do algoritmo

rnf_A.fit(x_treino,y_treino)



y_prever_rnf_A = rnf_A.predict(X_teste_A)



#Pontuação do algoritmo

print(rnf_A.score(x_treino,y_treino))



print('Accuracy score: {}'.format(accuracy_score(Y_teste_A, y_prever_rnf_A)))

print('Precision score: {}'.format(precision_score(Y_teste_A, y_prever_rnf_A)))

print('Recall score: {}'.format(recall_score(Y_teste_A, y_prever_rnf_A)))

print('F1 score: {}'.format(f1_score(Y_teste_A, y_prever_rnf_A)))
importancia = rnf_A.feature_importances_

print(importancia)

testeimpo_A = pd.DataFrame ({'Importancia':importancia,'Colunas':x_treino.columns})

print(testeimpo_A)

sns.set(rc={'figure.figsize':(18,12)})

sns.barplot(x=testeimpo_A['Importancia'],y=testeimpo_A['Colunas'],palette="Blues_d")
conf_matrix_A = confusion_matrix(Y_teste_A, y_prever_rnf_A)

print(pd.crosstab(Y_teste_A, y_prever_rnf_A, rownames=['Real'],

                  colnames=['Predito'], margins=True))
# Colunas para cada varíavel de teste/ Top 8

x_treino=treino[['Age', 'Fare', 'New_Sex','Pclass_3',

                 'Titulo_Mr','Titulo_Miss','Titulo_Mrs', 'Cabin']]

y_treino=treino['Survived']

x_teste=treino[['Age', 'Fare', 'New_Sex','Pclass_3',

                 'Titulo_Mr','Titulo_Miss','Titulo_Mrs', 'Cabin']]



x_treino, X_teste_8, y_treino, Y_teste_8 = train_test_split(treino[['Age', 'Fare', 'New_Sex','Pclass_3',

                 'Titulo_Mr','Titulo_Miss','Titulo_Mrs', 'Cabin']], treino['Survived'], random_state=1, test_size= 0.3)

#Algoritmo de Floresta aleatória

rnf_8=RandomForestClassifier()



#Aprendizado do algoritmo

rnf_8.fit(x_treino,y_treino)



#Predição do algoritmo

y_prever_rnf_8 = rnf_8.predict(X_teste_8)



#Pontuação do algoritmo

print(rnf_8.score(x_treino,y_treino))



print('Accuracy score: {}'.format(accuracy_score(Y_teste_8, y_prever_rnf_8)))

print('Precision score: {}'.format(precision_score(Y_teste_8, y_prever_rnf_8)))

print('Recall score: {}'.format(recall_score(Y_teste_8, y_prever_rnf_8)))

print('F1 score: {}'.format(f1_score(Y_teste_8, y_prever_rnf_8)))
importancia_8 = rnf_8.feature_importances_

print(importancia_8)

testeimpo_8 = pd.DataFrame ({'Importancia':importancia_8,'Colunas':x_treino.columns})

print(testeimpo_8)

sns.set(rc={'figure.figsize':(18,12)})

sns.barplot(x=testeimpo_8['Importancia'],y=testeimpo_8['Colunas'],palette="Blues_d")
conf_matrix_8 = confusion_matrix(Y_teste_8, y_prever_rnf_8)

print(pd.crosstab(Y_teste_8, y_prever_rnf_8, rownames=['Real'],

                  colnames=['Predito'], margins=True))
# Colunas para cada varíavel de teste/ Top 6

x_treino=treino[['Age', 'Fare', 'New_Sex','Pclass_3',

                 'Titulo_Mr','Cabin']]

y_treino=treino['Survived']

x_teste=treino[['Age', 'Fare', 'New_Sex','Pclass_3',

                 'Titulo_Mr','Cabin']]



x_treino, X_teste_6, y_treino, Y_teste_6 = train_test_split(treino[['Age', 'Fare', 'New_Sex','Pclass_3',

                 'Titulo_Mr','Cabin']], treino['Survived'], random_state=1)



#Algoritmo de Floresta aleatória

rnf_6=RandomForestClassifier()



#Aprendizado do algoritmo

rnf_6.fit(x_treino,y_treino)



#Predição do algoritmo

y_prever_rnf_6 = rnf_6.predict(X_teste_6)



#Pontuação do algoritmo

print(rnf_6.score(x_treino,y_treino))



print('Accuracy score: {}'.format(accuracy_score(Y_teste_6, y_prever_rnf_6)))

print('Precision score: {}'.format(precision_score(Y_teste_6, y_prever_rnf_6)))

print('Recall score: {}'.format(recall_score(Y_teste_6, y_prever_rnf_6)))

print('F1 score: {}'.format(f1_score(Y_teste_6, y_prever_rnf_6)))
importancia_6 = rnf_6.feature_importances_

print(importancia_6)

testeimpo_6 = pd.DataFrame ({'Importancia':importancia_6,'Colunas':x_treino.columns})

print(testeimpo_6)

sns.set(rc={'figure.figsize':(18,12)})

sns.barplot(x=testeimpo_6['Importancia'],y=testeimpo_6['Colunas'],palette="Blues_d")
conf_matrix_6 = confusion_matrix(Y_teste_6, y_prever_rnf_6)

print(pd.crosstab(Y_teste_6, y_prever_rnf_6, rownames=['Real'],

                  colnames=['Predito'], margins=True))
# Colunas para cada varíavel de treino teste final

x_treino_T=treino[['Age', 'Fare', 'New_Sex', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Pclass_1','Pclass_2','Pclass_3',

                 'Sozinho','Pequena_familia','Grande_familia','Titulo_Master','Titulo_Miss','Titulo_Mrs','Titulo_Mr',

                 'Cabin']]

y_treino_T=treino['Survived']

x_teste_T=treino[['Age', 'Fare', 'New_Sex', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Pclass_1','Pclass_2','Pclass_3',

                 'Sozinho','Pequena_familia','Grande_familia','Titulo_Master','Titulo_Miss','Titulo_Mrs','Titulo_Mr',

                 'Cabin']]



#Algoritmo de Floresta aleatória

rnf_T=RandomForestClassifier()



#Aprendizado do algoritmo

rnf_T.fit(x_treino_T, y_treino_T)



#Predição do algoritmo

y_prever_rnf_T = rnf_T.predict(x_teste_T)



#Pontuação do algoritmo

print(rnf_T.score(x_treino_T,y_treino_T))



print('Accuracy score: {}'.format(accuracy_score(y_treino_T, y_prever_rnf_T)))

print('Precision score: {}'.format(precision_score(y_treino_T, y_prever_rnf_T)))

print('Recall score: {}'.format(recall_score(y_treino_T, y_prever_rnf_T)))

print('F1 score: {}'.format(f1_score(y_treino_T, y_prever_rnf_T)))
print(pd.crosstab(y_treino_T, y_prever_rnf_T, rownames=['Real'],

                  colnames=['Predito'], margins=True))
# Colunas para cada varíavel de treino teste final

x_treino_T8=treino[['Age', 'Fare', 'New_Sex','Pclass_3',

                 'Titulo_Mr','Titulo_Miss','Titulo_Mrs', 'Cabin']]

y_treino_T8=treino['Survived']

x_teste_T8=treino[['Age', 'Fare', 'New_Sex','Pclass_3',

                 'Titulo_Mr','Titulo_Miss','Titulo_Mrs', 'Cabin']]



#Algoritmo de Floresta aleatória

rnf_T8=RandomForestClassifier()



#Aprendizado do algoritmo

rnf_T8.fit(x_treino_T8, y_treino_T8)



#Predição do algoritmo

y_prever_rnf_T8 = rnf_T8.predict(x_teste_T8)



#Pontuação do algoritmo

print(rnf_T8.score(x_treino_T8,y_treino_T8))



print('Accuracy score: {}'.format(accuracy_score(y_treino_T8, y_prever_rnf_T8)))

print('Precision score: {}'.format(precision_score(y_treino_T8, y_prever_rnf_T8)))

print('Recall score: {}'.format(recall_score(y_treino_T8, y_prever_rnf_T8)))

print('F1 score: {}'.format(f1_score(y_treino_T8, y_prever_rnf_T8)))
print(pd.crosstab(y_treino_T8, y_prever_rnf_T8, rownames=['Real'],

                  colnames=['Predito'], margins=True))
# Colunas para cada varíavel do modelo final

x_treino_F=treino[['Age', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Pclass_1','Pclass_2','Pclass_3',

                 'Sozinho','Pequena_familia','Grande_familia','Titulo_Master','Titulo_Miss','Titulo_Mrs','Titulo_Mr',

                 'Cabin']]

y_treino_F=treino['Survived']

x_teste_F=teste[['Age', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Pclass_1','Pclass_2','Pclass_3',

                 'Sozinho','Pequena_familia','Grande_familia','Titulo_Master','Titulo_Miss','Titulo_Mrs','Titulo_Mr',

                 'Cabin']]



#Algoritmo de Floresta aleatória

rnf_F=RandomForestClassifier()



#Aprendizado do algoritmo

rnf_F.fit(x_treino_F, y_treino_F)



#Predição do algoritmo

y_prever_rnf_F = rnf_F.predict(x_teste_F)



#Pontuação do algoritmo

print(rnf_F.score(x_treino_F, y_treino_F))
df_teste_rnf = pd.DataFrame({

        "PassengerId": teste["PassengerId"],

        "Survived": y_prever_rnf_F

         })