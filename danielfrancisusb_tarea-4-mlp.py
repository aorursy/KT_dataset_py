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
        self.prev_weights_diffs = [0]*dimension
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
            
print('Perceptron ready')
from math import e

def sigmoid(x):
    return (e**x)/(e**x+1)

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

print('Sigmoid ready')
class MLP():
    def __init__(self,num_attributes,entry_layer_size,
                 hidden_layer_size,exit_layer_size,
                 learning_rate,momentum):
        self.entry_layer_size = entry_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.exit_layer_size = exit_layer_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        self.entry_layer = []
        self.hidden_layer = []
        self.exit_layer = []
        
        for i in range(entry_layer_size):
            self.entry_layer += [Perceptron(num_attributes,(-0.05, 0.05),learning_rate)]
        
        for i in range(hidden_layer_size):
            self.hidden_layer += [Perceptron(self.entry_layer_size,(-0.05, 0.05),learning_rate)]
        
        for i in range(exit_layer_size):
            self.exit_layer += [Perceptron(self.hidden_layer_size,(-0.05, 0.05),learning_rate)]
        
    def feed_forward(self,x_vector):
        # x_vector -> entry_layer -> hidden_layer -> exit_layer
        self.entry_outputs = []
        self.hidden_outputs = []
        self.exit_outputs = []

        for neuron in self.entry_layer:
            neuron.latest_input = np.dot(neuron.weights,x_vector)
            self.entry_outputs += [neuron.latest_input]

        for neuron in self.hidden_layer:        
            neuron.latest_input = np.dot(neuron.weights,self.entry_outputs)
            self.hidden_outputs += [neuron.latest_input]
            
        for neuron in self.exit_layer:
            neuron.latest_input = np.dot(neuron.weights,self.hidden_outputs)
            self.exit_outputs +=  [neuron.latest_input]
        return self.exit_outputs
        
    def backpropagate(self,error_vector,target_vector):
        # entry_layer <- hidden_layer <- exit_layer
        alpha = self.momentum
        
        # exit_layer
        for j in range(len(self.exit_layer)):
            neuron = self.exit_layer[j]
            nu   = self.learning_rate                
            neuron.delta_vector = []
            e_j = error_vector[j]
            phi_j = sigmoid_derivative(np.dot(neuron.weights,self.hidden_outputs))
            delta_j = e_j*phi_j
            
            # Store delta for next step
            neuron.delta = delta_j
            
            for i in range(len(neuron.weights)):
                y_i = self.hidden_outputs[i]
                    
                # Calculate ∆W_ji(n) with momentum
                w_diff = nu*delta_j*y_i
                w_prev_diff = neuron.prev_weights_diffs[i]
                
                w_diff = alpha * w_prev_diff + nu*delta_j*y_i 
                
                # Update current weight
                neuron.weights[i] = neuron.weights[i] - w_diff 
                
                # Store w_div as previous
                neuron.prev_weights_diffs[i] = w_diff
        
        # hidden_layer
        for j in range(len(self.hidden_layer)):
            neuron = self.hidden_layer[j]
            nu   = self.learning_rate
            
            neuron.delta_vector = []
            phi_j = sigmoid_derivative(np.dot(neuron.weights,self.entry_outputs))          
            
            delta_vector = [neuron.delta for neuron in self.exit_layer]
            
            
            # 25, 10
            sigma = 0
            for a in range(len(self.exit_layer)):
                sigma += delta_vector[a]*self.exit_layer[a].weights[j]
            
            delta_j = phi_j*sigma
 
            for i in range(len(neuron.weights)):
                y_i = self.entry_outputs[j]
                
                # Calculate ∆W_ji(n) with momentum
                w_diff = nu*delta_j*y_i
                w_prev_diff = neuron.prev_weights_diffs[i]
                w_diff = alpha * w_prev_diff + nu*delta_j*y_i
                # Update current weight
                neuron.weights[i] = neuron.weights[i] - w_diff 
                
                # Store w_div as previous
                neuron.prev_weights_diffs[i] = w_diff

        return
        
        
print('MLP Ready')
# Se obtienen los datos suministrados a Kaggle

print('Loading training data...')
train_df = pd.read_csv('../input/mnist-usb/mnist_train.csv',header=None)

print(train_df.shape,'OK')
# Ejecución del algoritmo de perceptrón multicapa para
# distintas tasas de entrenamiento

from sklearn.utils import shuffle

def train_network(epochs,
                  num_samples,
                  hidden_layer_size,
                  learning_rate,
                  momentum):

    df_columns = train_df.shape[1]

    X = train_df.iloc[:num_samples,1:df_columns+1]/255
    X = np.array(X.values)
    num_attributes = X.shape[1]
    
    Y = np.array([y[0] for y in train_df.iloc[:num_samples,[0,]].values])

    mlp = MLP(num_attributes=num_attributes,
                   entry_layer_size=num_attributes,
                   hidden_layer_size=hidden_layer_size,
                   exit_layer_size=10,
                   learning_rate=learning_rate,
                   momentum=momentum)
    mlp.train_error_per_epoch = []
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
            
            
            layer_output = mlp.feed_forward(x_train)
            prediction = layer_output.index(max(layer_output))  
            
            
            error_vector = []
            for i in range(len(target_vector)):
                error_vector+=[layer_output[i] - target_vector[i]]
    
           
            validation = target_vector.index(max(target_vector))
            
            # Porque se quiere 0.9 para la unidad que representa 
            # la clase correcta y 0.1 en caso contrario, pero
            # puede haber más de una neurona de salida con valor
            # mayor a 0.9, vamos a escoger el valor máximo.
            
            if target_vector.index(max(target_vector)) == layer_output.index(max(layer_output)):
                score+=1
                
            mlp.backpropagate(error_vector,target_vector)
        
        mlp.train_error_per_epoch.append(abs(1-score/num_samples*100) )
        print('\tScore:',str(score/num_samples*100),'%')
        
    return mlp
print('Training algorithm ready')
# Uso del conjunto de prueba

print('Loading testing data...')
test_df = pd.read_csv('/kaggle/input/mnist-usb/mnist_test.csv',header=None)

print('OK')
# Ejecución del algoritmo de perceptrón multicapa para
# distintas tasas de entrenamiento

from sklearn.utils import shuffle

def test_network(model):

    df_columns = test_df.shape[1]
    num_samples = int((test_df.shape[0])/4)

    X = test_df.iloc[:num_samples,1:df_columns+1]/255
    X = np.array(X.values)
    num_attributes = X.shape[1]
    
    Y = np.array([y[0] for y in train_df.iloc[:num_samples,[0,]].values])

    mlp = model
    
    mlp.test_error_per_epoch = []

    print('Preparing algorithm')
    score = 0

    shuffler = np.random.permutation(len(X))
    X_shuffle = X[shuffler]
    Y_shuffle = Y[shuffler]

    for i in range(num_samples):
        if i%1000 == 0:
            print('\t\Testing sample #',i,'...')
        y_train = Y_shuffle[i]
        target_vector = [0]*10
        target_vector[y_train] = 1
        x_train = X_shuffle[i]

        layer_output = mlp.feed_forward(x_train)
        prediction = layer_output.index(max(layer_output))  

        error_vector = []
        for i in range(len(target_vector)):
            error_vector+=[layer_output[i] - target_vector[i]]


        validation = target_vector.index(max(target_vector))

        if target_vector.index(max(target_vector)) == layer_output.index(max(layer_output)):
            score+=1

        mlp.backpropagate(error_vector,target_vector)

    print('\tScore:',str(score/num_samples*100),'%')
    mlp.test_error_per_epoch.append(abs(1-score/num_samples*100))
    return mlp
print('Testing algorithm ready')
fix_epochs = 50
fix_samples = int(train_df.shape[0])

m_20 = train_network(
    epochs=2,
    num_samples=100,
    hidden_layer_size=20,
    learning_rate=0.1,
    momentum=0.9)

m_50 = train_network(
    epochs=2,
    num_samples=100,
    hidden_layer_size=50,
    learning_rate=0.1,
    momentum=0.9)

m_100 = train_network(
    epochs=2,
    num_samples=100,
    hidden_layer_size=100,
    learning_rate=0.1,
    momentum=0.9)

models = {'n20':m_20,'n50':m_50,'n100':m_100}
train_error_dict = {}
test_error_dict = {}

for key in models.keys():
    print(key)
    model = models[key]
    test_network(model)
    train_error_dict[key]= model.train_error_per_epoch
    test_error_dict[key]= model.test_error_per_epoch
print(list(range(len(train_error_dict['n20']))) )
from matplotlib import pyplot as plt

x_axis = list(range(len(train_error_dict['n20']))) 

models = {'n20':m_20,'n50':m_50,'n100':m_100}

plt.plot(x_axis, train_error_dict['n20'], label='Train n=20', color='blue',marker='H')
plt.plot(x_axis, train_error_dict['n50'], label='Train n=50', color='green')
plt.plot(x_axis, train_error_dict['n100'], label='Train n=100', color='red')

plt.xlabel("Epoca")
plt.ylabel("Precisión = |1-Error|")
plt.legend(loc='best')

plt.show()


print('Las puntuaciones de validación son:')
print(test_error_dict)
m_100_0 = train_network(
    epochs=2,
    num_samples=100,
    hidden_layer_size=100,
    learning_rate=0.1,
    momentum=0)

m_100_025 = train_network(
    epochs=2,
    num_samples=100,
    hidden_layer_size=100,
    learning_rate=0.1,
    momentum=0.25)

m_100_05 = train_network(
    epochs=2,
    num_samples=100,
    hidden_layer_size=100,
    learning_rate=0.1,
    momentum=0.5)

models['m_100_0']=m_100_0
models['m_100_025']=m_100_025
models['m_100_05']=m_100_05

train_error_dict['m_100_0']=m_100_0.train_error_per_epoch
train_error_dict['m_100_025']=m_100_025.train_error_per_epoch
train_error_dict['m_100_05']=m_100_05.train_error_per_epoch

test_error_dict['m_100_0']=m_100_0.test_error_per_epoch
test_error_dict['m_100_025']=m_100_025.test_error_per_epoch
test_error_dict['m_100_05']=m_100_05.test_error_per_epoch

print(train_error_dict)
print('Las nuevas puntuaciones de validación son:')
print(test_error_dict)
m_1_4 = train_network(
    epochs=2,
    num_samples=train_df.shape[0]/4,
    hidden_layer_size=100,
    learning_rate=0.1,
    momentum=0)

m_1_2 = train_network(
    epochs=2,
    num_samples=train_df.shape[0]/2,
    hidden_layer_size=100,
    learning_rate=0.1,
    momentum=0.25)

models['m_1_4']=m_1_4
models['m_1_2']=m_1_2

train_error_dict['m_1_4']=m_1_4.train_error_per_epoch
train_error_dict['m_1_2']=m_1_2.train_error_per_epoch

test_error_dict['m_1_4']=m_1_4.test_error_dict
test_error_dict['m_1_2']=m_1_2.test_error_dict


print(train_error_dict)
print('Las puntuaciones finales de validación son:')
print(test_error_dict)