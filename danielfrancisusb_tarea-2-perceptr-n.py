X = [[2,6],

     [1,3],

     [3,9]]



Y = [0,1,1]
# Utilizamos numpy y pandas para el manejo de los datos

import numpy as np 

import pandas as pd

import random

import math



# Se cargaron los archivos suministrados por la profesora a Kaggle

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Declaramos la clase Perceptron para modelar su comportamiento

class Perceptron():

    def __init__(self, dimension=1, interval=(0,1), learning_rate=1):

        self.weights = []

        for i in range(dimension):

            random.seed()

            self.weights.append(random.uniform(interval[0],interval[1]))

        self.input = [0]*dimension

        self.learning_rate = learning_rate

        self.output = 0



    def activate(self, weight_vector, x_vector, bias=0):

        weighted_sum = 0

        for i in range(len(weight_vector)-1):

            weighted_sum = weighted_sum + weight_vector[i] * x_vector[i]



#         Se solicita implementar la siguiente funcion de activacion,

#         pero obtenemos mejores resultados con la funcion lineal

#         print('Prev Output',self.output)

#         if weighted_sum > 0:

#             self.output = 1

#         else:

#             self.output = 0



        self.output = weighted_sum

        return self.output



    def update_weights(self,difference,learning_rate,x_vector):

        i = 0

        for i in range(len(self.weights)-1):

            self.weights[i] = self.weights[i] + learning_rate*difference*x_vector[i]

            



# Clase Layer o Capa, que agrupa distintos perceptrones.

class Layer():

    def __init__(self,num_neurons=1, input_length=1, interval=(0,1), learning_rate=0.1):

        self.neurons = [

            Perceptron(

                dimension=input_length,

                interval=interval,

                learning_rate=learning_rate

            ) for i in range(num_neurons)

        ]



    def __repr__(self):

        return(str("Layer of "+str(len(self.neurons))+" neurons"))

    

    print("Ready")
# Se obtienen los datos suministrados a Kaggle



print('Loading training data...')

train_df = pd.read_csv('/kaggle/input/mnist_train.csv',header=None)



print('OK')
# Ejecución del algoritmo de perceptrón para

# distintas tasas de entrenamiento



from sklearn.utils import shuffle



def train_network(learning_rate, num_samples, epochs):



    nn_layer = Layer(num_neurons=10, input_length=785, interval=(-0.05,0.05), learning_rate=0.01)



    num_attributes = train_df.shape[1]



    X = train_df.iloc[:num_samples,1:num_attributes+1]/255

    X = X.values

    Y = [y[0] for y in train_df.iloc[:num_samples,[0,]].values]



    X = np.array(X)

    Y = np.array(Y)



    print('Preparing algorithm')

    print('Using Learning Rate = ',learning_rate)

    for epoch in range(0,epochs):

        print('\tStarting Epoch',epoch)

        score = 0



        shuffler = np.random.permutation(len(X))

        X_shuffle = X[shuffler]

        Y_shuffle = Y[shuffler]



        for i in range(num_samples):

            if i%1000 == 0:

                print('\t\tTraining sample #',i,'...')

            y_train = Y_shuffle[i]

            target_vector = [0]*10

            target_vector[y_train] = 1

            x_train = X_shuffle[i]



            neuron_identifier = 0

            for neuron in nn_layer.neurons:

                output = neuron.activate(neuron.weights, list(x_train))

                difference = target_vector[neuron_identifier]-output

                neuron.update_weights(difference, learning_rate, list(x_train))

                neuron_identifier+=1



            layer_output = [neuron.output for neuron in nn_layer.neurons]

            prediction = layer_output.index(max(layer_output))



            validation = target_vector.index(max(target_vector))

#             print('Expected',target_vector.index(max(target_vector)),'Got',layer_output.index(max(layer_output)))

            if target_vector.index(max(target_vector)) == layer_output.index(max(layer_output)):

                score+=1



        print('\tScore:',str(score/num_samples*100),'%')

        

    return nn_layer
# Entrenamos los modelos con distintas tasas de aprendizaje

# No usaremos la totalidad de los datos para poder

# suministrar un resultado de forma breve.



learning_rates = [0.001, 0.01, 0.1]



model_dict = {}



for rate in learning_rates:

    model_dict[rate] = train_network(learning_rate=rate, num_samples=25000, epochs= 1)



# train_network(0.1,5000,3)
# Uso del conjunto de prueba



print('Loading testing data...')

test_df = pd.read_csv('/kaggle/input/mnist_test.csv',header=None)



print('OK')
def test_model(nn_layer,learning_rate):

    num_attributes = test_df.shape[1]

    num_samples = test_df.shape[0]



    X = test_df.iloc[:num_samples,1:num_attributes+1]/255

    X = X.values

    

    Y = [y[0] for y in test_df.iloc[:num_samples,[0,]].values]



    X = np.array(X)

    Y = np.array(Y)



    score = 0



    shuffler = np.random.permutation(len(X))

    X_shuffle = X[shuffler]

    Y_shuffle = Y[shuffler]



    for i in range(num_samples):

        if i%1000 == 0:

            print('\t\tTesting sample #',i,'...')

        y_train = Y_shuffle[i]

        target_vector = [0]*10

        target_vector[y_train] = 1

        x_train = X_shuffle[i]



        neuron_identifier = 0

        for neuron in nn_layer.neurons:

            output = neuron.activate(neuron.weights, list(x_train))

            difference = target_vector[neuron_identifier]-output

            neuron.update_weights(difference, learning_rate, list(x_train))

            neuron_identifier+=1



        layer_output = [neuron.output for neuron in nn_layer.neurons]

        prediction = layer_output.index(max(layer_output))



        validation = target_vector.index(max(target_vector))

#             print('Expected',target_vector.index(max(target_vector)),'Got',layer_output.index(max(layer_output)))

        if target_vector.index(max(target_vector)) == layer_output.index(max(layer_output)):

            score+=1



    print('\t Test score:',str(score/num_samples*100),'%')
test_model(model_dict[0.001],0.001)