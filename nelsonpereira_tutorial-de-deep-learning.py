import numpy as np # para algebra linear

import pandas as pd # para processamento de dados e arquivos .csv

import matplotlib.pyplot as plt # para visualização de dados em gráficos, etc

import warnings # para obter avisos do sistema



warnings.filterwarnings('ignore') #filtro de avisos



# apara poder obter os arquivos de dados do diretorio pai do Kernel e listar os mesmos

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# carregando e lendo o dataset com plotagem

x_l = np.load('../input/X.npy')

Y_l = np.load('../input/Y.npy')

img_size = 64

plt.subplot(1, 2, 1)

plt.imshow(x_l[260].reshape(img_size, img_size))

plt.axis('off')

plt.subplot(1,2,2)

plt.imshow(x_l[900].reshape(img_size, img_size))

plt.axis('off')
# Associa uma sequência de matrizes ao longo de um eixo de linha.

X = np.concatenate((x_l[204:409], x_l[822:1027]), axis=0) # de 0 a 204 é sinal de zero e de 205 a 410 é de um

z = np.zeros(205)

o = np.ones(205)

Y = np.concatenate((z,o), axis=0).reshape(X.shape[0],1)

print("Forma de X: ", X.shape)

print("Forma de Y: ", Y.shape)
# vamos criar matrizes x_train, y_train, x_test, y_test

from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

number_of_train = X_train.shape[0]

number_of_test = X_test.shape[0]
X_train_flatten = X_train.reshape(number_of_train, X_train.shape[1]*X_train.shape[2])

X_test_flatten = X_test.reshape(number_of_test, X_test.shape[1]*X_test.shape[2])

print("X train flatten", X_train_flatten.shape)

print("X teste flatten", X_test_flatten.shape)
x_train = X_train_flatten.T

x_test = X_test_flatten.T

y_train = Y_train.T

y_test = Y_test.T

print("x treino", x_train.shape)

print("x teste", x_test.shape)

print("y treino", y_train.shape)

print("y teste", y_test.shape)
# descrição curta e exemplo de definição (def)

def dummy(parameter):

    dummy_parameter = parameter + 5

    return dummy_parameter

result = dummy(3)
# permite inicializar parâmetros

# Então, precisamos da dimensão 4096, que é o número de pixels como parâmetro 

# para o nosso método de inicialização (def)



def initialize_weights_and_bias(dimension):

    w = np.full((dimension,1), 0.01)

    b = 0.0

    return w, b 

w,b = initialize_weights_and_bias(4096)
# calculo de z

#z = np.dot(w.T,x_train)+b

def sigmoid (z):

    y_head = 1/(1+np.exp(-z))

    return y_head



y_head = sigmoid(0)

y_head
# Etapas de propagação direta:

# encontrar z = w.T * x + b

# y_head = sigmoid (z)

# loss(error) = loss(y,y_head)

# cost = sum(loss)

def forward_propagation(w,b,x_train,y_train):

    z = np.dot(w.T, x_train) + b

    y_head = sigmoid(z) # probabilistic 0-1

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1] # x_train.shape [1] é para dimensionamento

    return cost
# Na propagação para trás, usaremos y_head que é encontrado na progressão para frente

# Portanto, em vez de escrever o método de propagação para trás, vamos combinar 

# propagação para frente e propagação para trás

def forward_backward_propagation (w,b,x_train,y_train):

    z = np.dot(w.T, x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss)) / x_train.shape[1] # x_train.shape [1] é para dimensionamento da     # propagação para trás

    derivate_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape [1] é      #para dimensionamento

    derivate_bias = np.sum(y_head-y_train) / x_train.shape[1]

    gradients = {"derivate_weight": derivate_weight, "derivate_bias": derivate_bias}

    return cost, gradients
# Atualizando parametros de aprendizagem

def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    for i in range(number_of_iterarion):

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        w = w - learning_rate * gradients["derivate_weight"]

        b = b - learning_rate * gradients["derivate_bias"]

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Custo após iteração %i: %f" %(i, cost))

    parameters = {"peso": w,"vies": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Numero de Iteração")

    plt.ylabel("Custo")

    plt.show()

    return parameters, gradients, cost_list
# predição

def predict(w,b,x_test):

    # x_test é uma entrada para propagação direta

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # se z for maior que 0,5, nossa previsão é o primeiro sinal (y_head = 1),

    # se z for menor que 0,5, nossa previsão é sinal zero (y_head = 0),

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # inicialização

    dimension =  x_train.shape[0]  # é 4096

    w,b = initialize_weights_and_bias(dimension)

    # não altere a taxa de aprendizado

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["peso"],parameters["vies"],x_test)

    y_prediction_train = predict(parameters["peso"],parameters["vies"],x_train)



    # Imprimir erros de treino / teste

    print("acuracia de treino: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("acuracia de teste: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 150)
from sklearn import linear_model

logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)

print("acuracia do teste: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))

print("acuracia do treino {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))
# inicializar parâmetros e tamanhos de camada

def initialize_parameters_and_layer_sizes_NN(x_train, y_train):

    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.1,

                  "bias1": np.zeros((3,1)),

                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1,

                  "bias2": np.zeros((y_train.shape[0],1))}

    return parameters
def forward_propagation_NN(x_train, parameters):

    Z1 = np.dot(parameters["weight1"], x_train) + parameters["bias1"]

    A1 = np.tanh(Z1)

    Z2 = np.dot(parameters["weight2"], A1) + parameters["bias2"]

    A2 = sigmoid(Z2)



    cache = {"Z1": Z1,

             "A1": A1,

             "Z2": Z2,

             "A2": A2}

    return A2, cache
# Custo de computação

def compute_cost_NN(A2, Y, parameters):

    logprobs =np.multiply(np.log(A2), Y)

    cost = -np.sum(logprobs)/ Y.shape[1]

    return cost
# Propagação para trás



def backward_propagation_NN(parameters, cache, X, Y):



    dZ2 = cache["A2"] - Y

    dW2 = np.dot(dZ2, cache["A1"].T) / X.shape[1]

    db2 = np.sum(dZ2, axis=1, keepdims=True) / X.shape[1]

    dZ1 = np.dot(parameters["weight2"].T, dZ2)*(1 - np.power(cache["A1"],2))

    dW1 = np.dot(dZ1,X.T) / X.shape[1]

    db1 = np.sum(dZ1, axis=1, keepdims=True) / X.shape[1]

    grads = {"dweight1": dW1,

             "dbias1": db1,

             "dweight2": dW2,

             "dbias2": db2}

    return grads 

# atualização de pArametros

def update_parameters_NN(parameters, grads, learning_rate = 0.01):

    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],

                  "bias1": parameters["bias1"]- learning_rate*grads["dbias1"],

                  "weight2" : parameters["weight2"] - learning_rate*grads["dweight2"],

                  "bias2": parameters["bias2"] - learning_rate*grads["dbias2"]}

    return parameters
# predição

def predict_NN(parameters,x_test):

    

    A2, cache = forward_propagation_NN(x_test,parameters)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    for i in range(A2.shape[1]):

        if A2[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction
def two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations):

    cost_list = []

    index_list = []

    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)



    for i in range(0, num_iterations):

        A2, cache = forward_propagation_NN(x_train,parameters)

        cost = compute_cost_NN(A2, y_train, parameters)

        grads = backward_propagation_NN(parameters, cache, x_train, y_train)

        parameters = update_parameters_NN(parameters, grads)



        if i % 100 == 0:

            cost_list.append(cost)

            index_list.append(i)

            print ("Custo depois de iteração %i: %f" %(i, cost))

    plt.plot(index_list,cost_list)

    plt.xticks(index_list,rotation='vertical')

    plt.xlabel("Numero de iterações")

    plt.ylabel("Custo")

    plt.show()



    y_prediction_test = predict_NN(parameters,x_test)

    y_prediction_train = predict_NN(parameters,x_train)



    print("acuracia de treino: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("acuracia de teste: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    return parameters



parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=2500)

# remodelar

x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T
# Implementando com a Biblioteca Keras

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential # inicializar biblioteca de rede neural

from keras.layers import Dense # contrução da  nossa biblioteca de camadas

def build_classifier():

    classifier = Sequential() # inicializar biblioteca de rede neuralk

    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))

    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)

mean = accuracies.mean()

variance = accuracies.std()

print("Acuracia da Média: "+ str(mean))

print("Accuracy da variação: "+ str(variance))