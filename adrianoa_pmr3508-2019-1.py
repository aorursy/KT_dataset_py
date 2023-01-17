%matplotlib inline

import pandas as pd

import numpy as np

import sklearn 

import os

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import seaborn as sns
#(abaixo o que implementei em meu computador)

#cwd = os.getcwd()

#os.listdir(cwd)
#base_file = (r"C:\Users\adria\Documents\POLI USP\7º Semestre\PMR3508\1_trabalho\train_data.csv")
#base_adult = pd.read_csv(base_file, 

                                         #names=[ "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status","Occupation", 

                                                        #"Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

                                                        #"Hours per week", "Country", "Target"],      

                                       # sep=r'\s*,\s*',

                                       # engine='python',

                                       # na_values = "?")
#test_file = (r"C:\Users\adria\Documents\POLI USP\7º Semestre\PMR3508\1_trabalho\test_data.csv")
#test_adult = pd.read_csv(test_file, 

                                         #names=[ 'Id', "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status","Occupation", 

                                                       # "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

                                                      #  "Hours per week", "Country", "Target"],      

                                        #sep=r'\s*,\s*',

                                       # engine='python',

                                       # na_values = "?")
#abaixo a adaptação que fiz para utilizar no Kaggle

base_adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

                         names=[ "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

                                                        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", 

                                                        "Capital Loss","Hours per week", "Country", "Target"], 

                        sep = r'\s*,\s*',

                        engine = 'python',

                        na_values = "?")



test_adult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

                         names=[ "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

                                                        "Occupation","Relationship", "Race", "Sex", "Capital Gain", 

                                                        "Capital Loss", "Hours per week", "Country", "Target"], 

                        sep = r'\s*,\s*',

                        engine = 'python',

                        na_values = "?")
base_adult.shape #Verificar quantos dados esou tratando
base_adult.head() #Verificar se a caracterização dos dados condiz com a ordem deles
base_adult["Age"].value_counts().plot(kind="pie")
base_adult["Workclass"].value_counts().plot(kind="bar")
base_adult["Education"].value_counts().plot(kind="bar")
base_adult["Education-Num"].value_counts().plot(kind="bar")
base_adult["Martial Status"].value_counts().plot(kind="bar")
base_adult["Occupation"].value_counts().plot(kind="bar")
base_adult["Relationship"].value_counts().plot(kind="bar")
base_adult["Race"].value_counts().plot(kind="bar")
base_adult["Sex"].value_counts().plot(kind="bar")
base_adult["Capital Gain"].value_counts().plot(kind="pie")
base_adult["Capital Loss"].value_counts().plot(kind="pie")
base_adult["Hours per week"].value_counts().plot(kind="pie")
base_adult["Country"].value_counts().plot(kind="bar")
base_adult["Target"].value_counts().plot(kind="bar")
#Nesta célula, preencheu-se os espaços em branco das variáveis que tinham sua caegoria com frequência maior que 50%  com essa maior frequência.

base_adult["Workclass"] = base_adult["Workclass"].fillna('Private')

base_adult["Race"] = base_adult["Race"].fillna('White')

base_adult["Sex"] = base_adult["Sex"].fillna('Male')

base_adult["Capital Gain"] = base_adult["Capital Gain"].fillna('0')

base_adult["Capital Loss"] = base_adult["Capital Loss"].fillna('0')

base_adult["Country"] = base_adult["Country"].fillna('United-States')

base_adult["Target"] = base_adult["Target"].fillna('<=50K ') 
#Aqui, após o preenchimento de alguns dados, elimina-se todas as linhas que tenham dados faltantes

nadult = base_adult.dropna()
nadult.shape #Para verificar o novo tamanho da base de teste
#Agora, tratarei cada uma das categorias, excluindo aquelas que não usarei (com a devida justificativa) e normalizando os dados em números

adult_treated = nadult



#Age -> Não fazer nada



#Workclass -> Agrupar os -gov. Agrupar os Self.

adult_treated= adult_treated.replace({'Private':0})

adult_treated= adult_treated.replace({'Self-emp-not-inc':1})

adult_treated= adult_treated.replace({'Local-gov':2})

adult_treated= adult_treated.replace({'State-gov':2})

adult_treated= adult_treated.replace({'Self-emp-inc':1})

adult_treated= adult_treated.replace({'Federal-gov':2})

adult_treated= adult_treated.replace({'Without-pay':3})



#fnlwgt -> não utilizar



#Education -> não utilizar

adult_treated = adult_treated.drop(['Education'], axis = 1)



#Education-Num -> Dado já tratado



#Matrial Status -> Agrupar Divorced e Separeted; Agrupar Married-civ-spouse com married-spouseabsent com married-AF-spouse; Widowe (3 categorias)

adult_treated= adult_treated.replace({'Married-civ-spouse':0})

adult_treated= adult_treated.replace({'Never-married':1})

adult_treated= adult_treated.replace({'Divorced':2})

adult_treated= adult_treated.replace({'Separated':2})

adult_treated= adult_treated.replace({'Widowed':2})

adult_treated= adult_treated.replace({'Married-spouse-absent':2})

adult_treated= adult_treated.replace({'Married-AF-spouse':2})



#Occupation -> Não utilizar

adult_treated = adult_treated.drop(['Occupation'], axis = 1)





#Relationship -> dividir entre quem tem filho e quem não tem, apenas

adult_treated= adult_treated.replace({'Husband':1})

adult_treated= adult_treated.replace({'Not-in-family':1})

adult_treated= adult_treated.replace({'Own-child':0})

adult_treated= adult_treated.replace({'Unmarried':1})

adult_treated= adult_treated.replace({'Wife':1})

adult_treated= adult_treated.replace({'Other-relative':1})



#Race ->  extratificar categorias, colocar um número para cada em ordem de mais densidade para menos densidade (não há dados que justifiquem organizar de outra forma)

adult_treated= adult_treated.replace({'White':0})

adult_treated= adult_treated.replace({'Black':1})

adult_treated= adult_treated.replace({'Asian-Pac-Islander':2})

adult_treated= adult_treated.replace({'Amer-Indian-Eskimo':3})

adult_treated= adult_treated.replace({'Other':4})



#Sex -> xtratificar categorias, colocar um número para cada em ordem de mais densidade para menos densidade

adult_treated= adult_treated.replace({'Male':0})

adult_treated= adult_treated.replace({'Female':1})



#Capital Gain e Capital Loss -> Excluir elemento 0 (outlier) e separar entre quem teve ganho e capital e quem não teve

adult_treated = adult_treated.drop("29256")



#Hour per Week -> já extratificado. Não fazer nada



#Country -> Separar entre estadunidenses(0), America (menos EUA + South(entendido como South Africa)) (3), Europa(1) e Ásia + Oriente Médio(2). O critério de divisao foi geográfico.

adult_treated= adult_treated.replace({'United-States':0})

adult_treated= adult_treated.replace({'Mexico':3})

adult_treated= adult_treated.replace({'Philippines':3})

adult_treated= adult_treated.replace({'Germany':1})

adult_treated= adult_treated.replace({'Puerto-Rico':3})

adult_treated= adult_treated.replace({'Canada':3})

adult_treated= adult_treated.replace({'El-Salvador':3})

adult_treated= adult_treated.replace({'India':2})

adult_treated= adult_treated.replace({'Cuba':3})

adult_treated= adult_treated.replace({'England':1})

adult_treated= adult_treated.replace({'Jamaica':3})

adult_treated= adult_treated.replace({'South':3})

adult_treated= adult_treated.replace({'China':2})

adult_treated= adult_treated.replace({'Italy':1})

adult_treated= adult_treated.replace({'Dominican-Republic':3})

adult_treated= adult_treated.replace({'Vietnam':2})

adult_treated= adult_treated.replace({'Guatemala':3})

adult_treated= adult_treated.replace({'Japan':2})

adult_treated= adult_treated.replace({'Poland':1})

adult_treated= adult_treated.replace({'Columbia':3})

adult_treated= adult_treated.replace({'Iran':2})

adult_treated= adult_treated.replace({'Taiwan':2})

adult_treated= adult_treated.replace({'Portugal':1})

adult_treated= adult_treated.replace({'Nicaragua':3})

adult_treated= adult_treated.replace({'Peru':3})

adult_treated= adult_treated.replace({'Greece':1})

adult_treated= adult_treated.replace({'Ecuador':3})

adult_treated= adult_treated.replace({'France':1})

adult_treated= adult_treated.replace({'Ireland':1})

adult_treated= adult_treated.replace({'Hong':2})

adult_treated= adult_treated.replace({'Trinadad&Tobago':3})

adult_treated= adult_treated.replace({'Cambodia':2})

adult_treated= adult_treated.replace({'Laos':2})

adult_treated= adult_treated.replace({'Thailand':2})

adult_treated= adult_treated.replace({'Yugoslavia':0})

adult_treated= adult_treated.replace({'Outlying-US(Guam-USVI-etc)':0})

adult_treated= adult_treated.replace({'Hungary':1})

adult_treated= adult_treated.replace({'Honduras':3})

adult_treated= adult_treated.replace({'Scotland':1})

adult_treated= adult_treated.replace({'Holand-Netherlands':1})

adult_treated= adult_treated.replace({'Haiti':3})



#Target

adult_parametro = adult_treated

adult_parametro = adult_parametro.replace({'<=50K': 0})

adult_parametro = adult_parametro.replace({'>50K': 1})
adult_treated = adult_treated.drop(adult_treated.index[0]) #Aqui elimina-se a linha "intrusa" observada acima
#Agora comecei a repetir o tratamento dado na base de aprendizado na base objeto

test_adult["Workclass"] = test_adult["Workclass"].fillna('Private')

test_adult["Race"] = test_adult["Race"].fillna('White')

test_adult["Sex"] = test_adult["Sex"].fillna('Male')

test_adult["Capital Gain"] = test_adult["Capital Gain"].fillna(0)

test_adult["Capital Loss"] = test_adult["Capital Loss"].fillna(0)

test_adult["Country"] = test_adult["Country"].fillna('United-States')

test_adult["Target"] = test_adult["Target"].fillna('<=50K ') 
test_adult.shape
test_treated
#Agora, tratarei cada uma das categorias, excluindo aquelas que não usarei (com a devida justificativa) e normalizando os dados em números

test_treated = test_adult



#Age -> Não fazer nada



#Workclass -> Agrupar os -gov. Agrupar os Self.

test_treated= test_treated.replace({'Private':0})

test_treated= test_treated.replace({'Self-emp-not-inc':1})

test_treated= test_treated.replace({'Local-gov':2})

test_treated= test_treated.replace({'State-gov':2})

test_treated= test_treated.replace({'Self-emp-inc':1})

test_treated= test_treated.replace({'Federal-gov':2})

test_treated= test_treated.replace({'Without-pay':3})



#fnlwgt -> não utilizar



#Education -> não utilizar



#Education-Num -> Dado já tratado



#Matrial Status -> Agrupar Divorced e Separeted; Agrupar Married-civ-spouse com married-spouseabsent com married-AF-spouse; Widowe (3 categorias)

test_treated= test_treated.replace({'Married-civ-spouse':0})

test_treated= test_treated.replace({'Never-married':1})

test_treated= test_treated.replace({'Divorced':2})

test_treated= test_treated.replace({'Separated':2})

test_treated= test_treated.replace({'Widowed':2})

test_treated= test_treated.replace({'Married-spouse-absent':2})

test_treated= test_treated.replace({'Married-AF-spouse':2})



#Occupation -> Não utilizar



#Relationship -> dividir entre quem tem filho e quem não tem, apenas

test_treated= test_treated.replace({'Husband':1})

test_treated= test_treated.replace({'Not-in-family':1})

test_treated= test_treated.replace({'Own-child':0})

test_treated= test_treated.replace({'Unmarried':1})

test_treated= test_treated.replace({'Wife':1})

test_treated= test_treated.replace({'Other-relative':1})



#Race ->  extratificar categorias, colocar um número para cada em ordem de mais densidade para menos densidade (não há dados que justifiquem organizar de outra forma)

test_treated= test_treated.replace({'White':0})

test_treated= test_treated.replace({'Black':1})

test_treated= test_treated.replace({'Asian-Pac-Islander':2})

test_treated= test_treated.replace({'Amer-Indian-Eskimo':3})

test_treated= test_treated.replace({'Other':4})



#Sex -> xtratificar categorias, colocar um número para cada em ordem de mais densidade para menos densidade

test_treated= test_treated.replace({'Male':0})

test_treated= test_treated.replace({'Female':1})



#Capital Gain e Capital Loss -> Excluir outliers (se der tempo)



#Hour per Week -> já extratificado. Não fazer nada



#Country -> Separar entre estadunidenses(0), America (menos EUA + South(entendido como South Africa)) (3), Europa(1) e Ásia + Oriente Médio(2). O critério de divisao foi geográfico.

test_treated= test_treated.replace({'United-States':0})

test_treated= test_treated.replace({'Mexico':3})

test_treated= test_treated.replace({'Philippines':3})

test_treated= test_treated.replace({'Germany':1})

test_treated= test_treated.replace({'Puerto-Rico':3})

test_treated= test_treated.replace({'Canada':3})

test_treated= test_treated.replace({'El-Salvador':3})

test_treated= test_treated.replace({'India':2})

test_treated= test_treated.replace({'Cuba':3})

test_treated= test_treated.replace({'England':1})

test_treated= test_treated.replace({'Jamaica':3})

test_treated= test_treated.replace({'South':3})

test_treated= test_treated.replace({'China':2})

test_treated= test_treated.replace({'Italy':1})

test_treated= test_treated.replace({'Dominican-Republic':3})

test_treated= test_treated.replace({'Vietnam':2})

test_treated= test_treated.replace({'Guatemala':3})

test_treated= test_treated.replace({'Japan':2})

test_treated= test_treated.replace({'Poland':1})

test_treated= test_treated.replace({'Columbia':3})

test_treated= test_treated.replace({'Iran':2})

test_treated= test_treated.replace({'Taiwan':2})

test_treated= test_treated.replace({'Portugal':1})

test_treated= test_treated.replace({'Nicaragua':3})

test_treated= test_treated.replace({'Peru':3})

test_treated= test_treated.replace({'Greece':1})

test_treated= test_treated.replace({'Ecuador':3})

test_treated= test_treated.replace({'France':1})

test_treated= test_treated.replace({'Ireland':1})

test_treated= test_treated.replace({'Hong':2})

test_treated= test_treated.replace({'Trinadad&Tobago':3})

test_treated= test_treated.replace({'Cambodia':2})

test_treated= test_treated.replace({'Laos':2})

test_treated= test_treated.replace({'Thailand':2})

test_treated= test_treated.replace({'Yugoslavia':0})

test_treated= test_treated.replace({'Outlying-US(Guam-USVI-etc)':0})

test_treated= test_treated.replace({'Hungary':1})

test_treated= test_treated.replace({'Honduras':3})

test_treated= test_treated.replace({'Scotland':1})

test_treated= test_treated.replace({'Holand-Netherlands':1})

test_treated= test_treated.replace({'Haiti':3})



#Target

adult_parametro = adult_treated

adult_parametro = adult_parametro.replace({'<=50K': 0})

adult_parametro = adult_parametro.replace({'>50K': 1})
test_treated = test_treated.drop(test_treated.index[0]) #Aqui elimina-se a linha "intrusa" observada acima
test_treated.shape
aux = adult_parametro.astype(np.int)



corr_mat = aux.corr()

f, ax = plt.subplots(figsize=(20, 13))

sns.heatmap(corr_mat, vmax=.7, square=True, cmap="coolwarm", annot = True)
Xadult = adult_treated[["Age", "Martial Status", "Relationship","Sex", "Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

Yadult = adult_treated.Target



XtestAdult = test_treated[["Age", "Martial Status", "Relationship", "Sex", "Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

YtestAdult = test_treated.Target
Yadult
i = 1

x = 0

#Essa etapa demora aproximadamente 4min

#Essa etapa tem a função de otimizar o parâmetro K. Implementa-se aqui o algorítimo requerido no enunciado para k entre 1 e 30, para 

while i <31:

    knn = KNeighborsClassifier(n_neighbors=i)

    scores = cross_val_score(knn, Xadult, Yadult, cv=10)

    knn.fit(Xadult,Yadult)

    YtestPred = knn.predict(XtestAdult)

    x_novo = accuracy_score(YtestAdult,YtestPred)

    if x_novo > x:

        x = x_novo

    i += 1
scores
id_index = pd.DataFrame({'Id' : list(range(len(YtestPred)))})

income = pd.DataFrame({'income' : YtestPred})

result = income

result
result.to_csv("submission-Adriano.csv", index = True, index_label = 'Id') #Gravação do arquivo para envio