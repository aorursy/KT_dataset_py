#Realize uma analise comparativa entre os algoritmos KNN, Árvores de
#Decisão e MLP, para classificar o tipo de vinho utilizando a base de dados
#encontrada em: https://www.kaggle.com/brynja/wineuci
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn import tree

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
dados_vinho = pd.read_csv('../input/wineuci/Wine.csv', header=None)
dados_vinho.head()
dados_vinho.columns = ['class', 'alcohol', 'malicAcid', 'ash', 'alcalinityofash', 'magnesium', 'totalPhenols',
                      'flavanoids', 'nonFlavanoidPhenols', 'proanthocyanins', 'colorIntensity', 'hue', 
                      'od280_od315', 'proline']
dados_vinho.head()
dados_vinho['class'].unique()
dados_vinho.describe()
dados_vinho.isnull().sum()
dados_vinho.info()
dados_vinho['class'].value_counts()
X = dados_vinho.iloc[:,:-1].values
X
y = dados_vinho.iloc[:,-0]
y
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.25, random_state = 0
)
y_train
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean')
knn.fit(X,y)
knn
knn.classes_
y_pred = knn.predict(X_train)
acc = metrics.accuracy_score(y_train, y_pred)
print(acc)
y_pred = knn.predict(X_test)
X_test[0]
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
x = np.array([  1.  ,  13.74,   1.67,   2.25,  16.4 , 118.  ,   2.6 ,   2.9 ,
         0.21,   1.62,   5.85,   0.92,   3.2 ])
knn.predict(X)
knn.predict(x.reshape(1, -1))
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    X, y, test_size = 0.25, random_state = 0
)
y_train_1
cart = DecisionTreeClassifier(max_depth = 3, criterion = "entropy")
cart.fit(X,y)
tree.plot_tree(cart)
cart.classes_
y_pred_1 = cart.predict(X_train_1)
acc_1 = metrics.accuracy_score(y_train_1, y_pred_1)
print(acc_1)
y_pred_1 = cart.predict(X_test_1)
acc_1 = metrics.accuracy_score(y_test_1, y_pred_1)
print(acc_1)
X_test_1[0]
x = np.array([1.  ,  13.74,   1.67,   2.25,  16.4 , 118.  ,   2.6 ,   2.9 ,
         0.21,   1.62,   5.85,   0.92,   3.2])
cart.predict(X)
cart.predict(x.reshape(1, -1))
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X, y, test_size = 0.65, random_state = 2
)
y_train_2
neurons = 8
functionAct = 'identity'
train = 'sgd'
learningrate = 0.02
mlp = MLPClassifier(hidden_layer_sizes=neurons,  activation=functionAct, 
                    solver=train, learning_rate_init=learningrate)
mlp
mlp.fit(X,y)
mlp.classes_
y_pred_2 = mlp.predict(X_train_2)
acc_2 = metrics.accuracy_score(y_train_2, y_pred_2)
print(acc_2)
X_test_2
y_pred_2 = mlp.predict(X_test_2)
acc_2 = metrics.accuracy_score(y_test_2, y_pred_2)
print(acc_2)
X_test_2[0]
x = np.array([ 1.  , 13.75,  1.73,  2.41, 16.  , 89.  ,  2.6 ,  2.76,  0.29,
        1.81,  5.6 ,  1.15,  2.9 ])
mlp.predict(x.reshape(1, -1))
# KNN - algoritmo k- vizinhos mais próximos é um método não paramétrico usado para classificação e regressão
# Caso tenha uma grande base de dados, será necessário realizar uma realimentação no modelo também em registros,
# ou seja, tornará a atividade computacional custosa. Na base de dados Wine o KNN teve acuracia de 94%  com os dados
# de treino e 93% com os testes
#------------------------------------------------------------------------------------------------------------------#
# Arvore de Decisão - são métodos de aprendizado de máquinas supervisionado não-paramétricos, muito utilizados em 
# tarefas de classificação e regressão. Dependendo de sua aprofundidade será facil de compreender suas especificações.
#Defini a profundidade em 3 obtive 95% acuracia com os dados de treino e 100% com os teste.
#-------------------------------------------------------------------------------------------------------------------#
# MLP – è um modelo de aprendizado supervisionado onde é fornecido uma amostra de dados informando a classificação
# dos mesmo, assim treinando o modelo para que quando uma nova amostra foi recebedo o mesmo seja capaz de estabelecer
# a que grupo uma amostra pertence. Usando os dados de treino o modelo alcançou uma taxa de 45% de acuracia e 38% com
# os dados de teste. 
# Observando que a quandtidade de neurônios influência diretamente na precisão do modelo.
#-------------------------------------------------------------------------------------------------------------------#
# Após a analise verifiquei que o modelo de Arvore de Decisão obteve o melhor resultados entre todos os modelos.