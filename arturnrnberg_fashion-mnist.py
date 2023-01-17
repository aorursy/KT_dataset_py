import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



import sklearn.neural_network

import sklearn.model_selection

import sklearn.metrics



from tensorflow import keras

from random import choices



from sklearn.model_selection import train_test_split



%matplotlib inline
data = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
X, y = data.iloc[:,1:].values/255, data.iloc[:,0].values
N_images = 50



# localização dos exemplos na matriz de dados 

rows = choices(range(0, 60000), k=N_images)



# selecionando os dígitos, já no formato de matriz

digitos = [X[i].reshape(28,28) for i in rows]

label = y[rows]



# criando figura do matplotlib

fig, ax = plt.subplots(5, int(len(rows)/5),figsize=(18,10))



# plotando!

for i in range(len(rows)):

    j = int(i/10)

    k = i - j*10

    ax[j, k].imshow(digitos[i], cmap = matplotlib.cm.binary, interpolation="nearest")

    ax[j, k].set_title(label[i])

    ax[j, k].axis('off')

label = {

0:'T-shirt/top',

1:'Trouser',

2:'Pullover',

3:'Dress',

4:'Coat',

5:'Sandal',

6:'Shirt',

7:'Sneaker',

8:'Bag',

9:'Ankle boot'

}
X_treino, X_validacao, y_treino, y_validacao = train_test_split(X, y, test_size=0.2, random_state=0)
fig, ax = plt.subplots(1,2,figsize=(13,4))



sns.countplot(y_treino,ax=ax[0])

sns.countplot(y_validacao,ax=ax[1])



ax[0].set_title('Treino')

ax[1].set_title('Validação')



fig.suptitle('Proporção de classes (roupas)');
m = sklearn.neural_network.MLPClassifier(random_state=0)



%time m.fit(X_treino, y_treino)
def acuracia(modelo, X_treino, X_validacao):



    acc_tr = sklearn.metrics.accuracy_score(y_treino, modelo.predict(X_treino))

    acc_val = sklearn.metrics.accuracy_score(y_validacao, modelo.predict(X_validacao))



    return {'Acurácia do treino': acc_tr, 'Acurácia da validação': acc_val}
acuracia(m, X_treino, X_validacao)
def compare_confusion_matriz(modelo, X_treino, X_validacao):

    

    y_validacao_pred = modelo.predict(X_validacao)

    y_train_pred = modelo.predict(X_treino)

    confusao_val = sklearn.metrics.confusion_matrix(y_validacao, y_validacao_pred)

    confusao_tr = sklearn.metrics.confusion_matrix(y_treino, y_train_pred)

    

    fig, ax = plt.subplots(1, 2,figsize=(20,10))

    sns.heatmap(pd.DataFrame(confusao_val).rename(label).rename(columns=label), ax=ax[0], cbar=False)

    ax[0].set_title('Matriz de confusão validação', size=20)

    ax[0].set_yticklabels(ax[0].get_xticklabels(), rotation=0, size=15)

    ax[0].set_xticklabels(ax[0].get_yticklabels(), rotation=90, size=15)

    sns.heatmap(pd.DataFrame(confusao_tr).rename(label).rename(columns=label), ax=ax[1], cbar=False)

    ax[1].set_title('Matriz de confusão treino', size=20)

    ax[1].set_yticklabels(ax[1].get_xticklabels(), rotation=0, size=15)

    ax[1].set_xticklabels(ax[1].get_yticklabels(), rotation=90, size=15)

    plt.show()
compare_confusion_matriz(m, X_treino, X_validacao)
modelo_ajustado = sklearn.neural_network.MLPClassifier(random_state=0,

                                                        hidden_layer_sizes=(400,),

                                                      )



modelo_ajustado.fit(X_treino, y_treino)



acuracia_400 = acuracia(modelo_ajustado, X_treino, X_validacao)

acuracia_400
modelo_ajustado = sklearn.neural_network.MLPClassifier(random_state=0,

                                                        hidden_layer_sizes=(100, 50),

                                                      )



modelo_ajustado.fit(X_treino, y_treino)



acuracia_100_50 = acuracia(modelo_ajustado, X_treino, X_validacao)

acuracia_100_50
funcoes_ativacao = ['identity', 'logistic', 'tanh']

acuracia_funcao = dict()



for function in funcoes_ativacao:

    modelo_ajustado = sklearn.neural_network.MLPClassifier(random_state=0,

                                                        hidden_layer_sizes=(100, 50),

                                                        activation = function

                                                      )



    modelo_ajustado.fit(X_treino, y_treino)



    acuracia_funcao[function] = acuracia(modelo_ajustado, X_treino, X_validacao)



acuracia_funcao
momemnto_list = [0.3,0.5, 0.7]

acuracia_momento = dict()



for momento in momemnto_list:

    modelo_ajustado = sklearn.neural_network.MLPClassifier(random_state=0,

                                                        hidden_layer_sizes=(100, 50),

                                                        momentum = momento

                                                      )



    modelo_ajustado.fit(X_treino, y_treino)



    acuracia_momento[momento] = acuracia(modelo_ajustado, X_treino, X_validacao)



acuracia_momento
batch_size_list = [32, 64, 128]

acuracia_batch_size = dict()



for batch_size in batch_size_list:

    modelo_ajustado = sklearn.neural_network.MLPClassifier(random_state=0,

                                                        hidden_layer_sizes=(100, 50),

                                                        batch_size = batch_size

                                                      )



    modelo_ajustado.fit(X_treino, y_treino)



    acuracia_batch_size[batch_size] = acuracia(modelo_ajustado, X_treino, X_validacao)



acuracia_batch_size
from math import cos, sin, radians, asin
constante1 = 3/360
k_list = range(1, 720, 2)





label0 = pd.DataFrame(data = {'X': [i*constante1*cos(radians(i)) for i in k_list],

                              'Y': [i*constante1*sin(radians(i)) for i in k_list],

                              'label': 0})



label1 = pd.DataFrame(data = {'X': [-i*constante1*cos(radians(i)) for i in k_list],

                              'Y': [-i*constante1*sin(radians(i)) for i in k_list],

                              'label': 1})
data = pd.concat([label0, label1])
sns.scatterplot(x=data.X, y=data.Y, hue=data.label, alpha=0.4)

plt.show()
data['raio'] = (data.X**2 + data.Y**2)**0.5
data['angulo'] = (data.Y/data.raio).map(asin)
sns.scatterplot(x=data.raio, y=data.angulo, hue=data.label, alpha=0.4)

plt.show()
def convert_angulo(raio, angulo):

    periodo = 1.5

    sinal = -1

    

    if ((raio>periodo) & (raio<2*periodo)) or (raio>3*periodo):

        sinal = 1

    

    return angulo*sinal
data['angulo_transformado'] = data.loc[:, ['raio', 'angulo']].apply(lambda x: convert_angulo(raio=x[0], angulo=x[1]), axis=1)
sns.scatterplot(x=data.raio, y=data.angulo_transformado, hue=data.label, alpha=0.4)

plt.show()
data['label_separado'] = [1 if i>0 else 0 for i in data.angulo_transformado]
sns.scatterplot(x=data.raio, y=data.label_separado, hue=data.label, alpha=0.4)

plt.show()
data[data.label != data.label_separado]