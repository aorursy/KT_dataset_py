

import pandas as pd #Análise de Dados

import numpy as np #Algebra Linear

import matplotlib.pyplot as plt #Gráficos

import seaborn as sns # Gráficos

from sklearn.metrics import mean_absolute_error #Calculo de Erro de Média Absoluta

from sklearn.model_selection import train_test_split #Divisão de dados em Treino e Validação

from sklearn.tree import DecisionTreeRegressor #Modelo de Arvore de decisão de regressão

from sklearn.tree import DecisionTreeClassifier #Modelo de Arvore de decisão de classificação

from sklearn.ensemble import RandomForestClassifier #Modelo de Floresta Aleatória

from sklearn import svm #Modelo SVM

from sklearn import metrics #Calcular Precisão do Modelo
pulsar_file_path = '../input/predicting-a-pulsar-star/pulsar_stars.csv'

pulsar_data = pd.read_csv(pulsar_file_path, header=0)

pulsar_data.info()
pulsar_data.head()
pulsar_data.describe()
plt.style.use('fivethirtyeight')

sns.catplot(x="target_class", kind="count", palette="ch:.25", data=pulsar_data, height=6);

print("Porcentagem de Estrelas Não Pulsares:   ", 

      round(pulsar_data["target_class"].value_counts()[0]/len(pulsar_data) * 100, 2), "%")

print("Porcentagem de Estrelas Pulsares:       ", 

      round(pulsar_data["target_class"].value_counts()[1]/len(pulsar_data) * 100, 2), "%")
#Primeiro vou testar o modelo com todas as colunas

#Retiro apenas a coluna target usando a função drop

X = pulsar_data.drop(['target_class'], axis=1)



#A variável y recebe a coluna target 

y = pulsar_data['target_class']



#Dividindo os dados 

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
X.isnull().mean()
#Definindo número de folhas com menor número de Erro de Média Absoluta

#Função que o obtem o valor de Erro de Média Absoluta para cada número de camadas

def get_acc(n_estimators, train_X, val_X, train_y, val_y):

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=1)

    model.fit(train_X, train_y)

    val_predictions = model.predict(val_X)

    accuracy = metrics.accuracy_score(val_predictions,val_y)

    return(accuracy)



#Loop com número de camadas 

for candidate_n_estimators in [5, 10, 25, 50, 100, 250, 500]:

    my_acc = get_acc(candidate_n_estimators, train_X, val_X, train_y, val_y)

    print('Número de Folhas:%d \t\t\t Precisão: %f' %(candidate_n_estimators, my_acc))
#Criando modelo com número de folhas escolhido acima

rf_model = RandomForestClassifier(n_estimators=100, random_state=1)

rf_model.fit(train_X, train_y)

val_predictions = rf_model.predict(val_X)



#Medindo a precisão

randomforest_acc = metrics.accuracy_score(val_predictions,val_y)

print('Precisão do Modelo:  ' + str(randomforest_acc))
#Definindo número de folhas com menor número de Erro de Média Absoluta

#Função que o obtem o valor de Erro de Média Absoluta para cada número de camadas

def get_acc(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=1)

    model.fit(train_X, train_y)

    val_predictions = model.predict(val_X)

    accuracy = metrics.accuracy_score(val_predictions,val_y)

    return(accuracy)



#Loop com número de camadas 

for candidate_max_leaf_nodes in [5, 10, 25, 50, 100, 250, 500]:

    my_acc = get_acc(candidate_max_leaf_nodes, train_X, val_X, train_y, val_y)

    print('Número de Folhas:%d \t\t\t Precisão: %f' %(candidate_max_leaf_nodes, my_acc))
#Criando modelo com número de folhas escolhido acima

ad_model = DecisionTreeClassifier(max_leaf_nodes = 50, random_state=1)

ad_model.fit(train_X, train_y)

val_predictions = ad_model.predict(val_X)



decisiontree_acc = metrics.accuracy_score(val_predictions,val_y)

print('Precisão do Modelo:  ' + str(decisiontree_acc))
svm_model = svm.SVC()

svm_model.fit(train_X,train_y)

val_predictions=svm_model.predict(val_X)

svm_acc = metrics.accuracy_score(val_predictions,val_y)

print('Precisão do Modelo:  ' + str(svm_acc))