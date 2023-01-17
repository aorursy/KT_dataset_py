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



class Adaline():

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

        for i in range(len(weight_vector)):

            weighted_sum = weighted_sum + weight_vector[i] * x_vector[i]

        self.output = weighted_sum

        return self.output



    def update_weights(self,w_difference):

        i = 0

        for i in range(len(self.weights)):

            self.weights[i] = self.weights[i]+w_difference[i]

            



# Clase Layer o Capa, que agrupa distintos adaline.

class Layer():

    def __init__(self,num_neurons=1, input_length=1, interval=(0,1), learning_rate=0.1):

        self.neurons = [

            Adaline(

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

train_df = pd.read_csv('../input/mnist-usb/mnist_train.csv',header=None)



print(train_df.shape,'OK')
# Necesitamos una función para el descenso de gradiente



def gradient_descent(x_input,difference):

    

    gradient_vector = []

    for k in range(len(x_input)):

        gradient_vector += [difference*x_input[k]]

        

    return gradient_vector
# Ejecución del algoritmo de perceptrón para

# distintas tasas de entrenamiento



from sklearn.utils import shuffle



def train_network(learning_rate, num_samples, epochs):



    nn_layer = Layer(num_neurons=10, input_length=784, interval=(-0.05,0.05), learning_rate=0.01)



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



            for neuron in nn_layer.neurons:

                output = neuron.activate(neuron.weights, list(x_train))



            layer_output = [neuron.output for neuron in nn_layer.neurons]

            prediction = layer_output.index(max(layer_output))

            

            difference_vector=[]

            for k in range(len(layer_output)):

                difference_vector += [target_vector[k] - layer_output[k]]

        

            neuron_identifier = 0

            for neuron in nn_layer.neurons:

                gradient = gradient_descent(list(x_train),difference_vector[neuron_identifier])

                w_difference = [g*learning_rate for g in gradient]

                neuron.update_weights(w_difference) 

                neuron_identifier+=1

            

            validation = target_vector.index(max(target_vector))



            if target_vector.index(max(target_vector)) == layer_output.index(max(layer_output)):

                score+=1



        print('\tScore:',str(score/num_samples*100),'%')

        

    return nn_layer
# Entrenamos los modelos con distintas tasas de aprendizaje



learning_rates = [0.001, 0.01, 0.1]



model_dict = {}





for rate in learning_rates:

    model_dict[rate] = train_network(learning_rate=rate, num_samples=25000, epochs= 1)



# train_network(0.001,10000,3)
# Uso del conjunto de prueba



print('Loading testing data...')

test_df = pd.read_csv('/kaggle/input/mnist-usb/mnist_test.csv',header=None)



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



        for neuron in nn_layer.neurons:

            output = neuron.activate(neuron.weights, list(x_train))



        layer_output = [neuron.output for neuron in nn_layer.neurons]

        prediction = layer_output.index(max(layer_output))



        difference_vector=[]

        for k in range(len(layer_output)):

            difference_vector += [target_vector[k] - layer_output[k]]



        neuron_identifier = 0

        for neuron in nn_layer.neurons:

            gradient = gradient_descent(list(x_train),difference_vector[neuron_identifier])

            w_difference = [g*learning_rate for g in gradient]

            neuron.update_weights(w_difference) 

            neuron_identifier+=1



        validation = target_vector.index(max(target_vector))        

        

        if target_vector.index(max(target_vector)) == layer_output.index(max(layer_output)):

            score+=1



    print('\t Test score:',str(score/num_samples*100),'%')
test_model(model_dict[0.001],0.001)
# Cargamos los datos suministrados para conseguir un interpolador



print('Loading interpolation data...')

interpolation_df = pd.read_csv('../input/interpolador-adaline/datosT3 - datosT3.csv',header=None)



print('OK')
interpolation_df.head()

print(interpolation_df.shape)
from sklearn.model_selection import train_test_split



X=interpolation_df.iloc[:,0].values

X=X/max(X)

y=interpolation_df.iloc[:,1].values

y=y/max(y)

learning_rates = [0.001,0.01,0.1]



epochs = 25



for rate in learning_rates:

    neuron = Adaline(dimension=1)

    for e in range(epochs):

        print("Training with rate",rate)

        index = 0

        score = 0

        for sample in X:

            prediction = neuron.activate(neuron.weights, [sample])        

            difference = y[index] - prediction

            

                

            gradient = gradient_descent([sample],difference)

            if sample == -2.0:

                print(gradient,sample,difference)

            w_difference = [g*rate for g in gradient]



            neuron.update_weights(w_difference)



            validation = y[index]



            if str(validation)[1:4] == str(prediction)[1:4]:

                #print(validation,prediction)

                score+=1



            index += 1



    print('\t Training score:',str(score/len(X)*100),'%. Interpolator =',neuron.weights)