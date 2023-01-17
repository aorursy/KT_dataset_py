import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rnd
from random import randint


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dss = pd.read_csv("../input/derm.csv")
dss.describe()
dss.drop(['age'], axis=1, inplace=True)
last_col = dss.columns[len(dss.columns)-1]
classes = list(dss[last_col].unique())
len_cols = len(dss.columns) - 1
# One Hot Codification
# Coding using zeros' array and 1 to each class
def one_hot_encoding(classes):
    cl_onehot = np.zeros((len(classes),len(classes)),dtype=int)
    np.fill_diagonal(cl_onehot,1)
    r = [(classes[i], cl) for i, cl in enumerate(cl_onehot)]
    return r

# Encode the expected classes
def encode_expected(expected, encoded_class):
    return np.array([ list(filter(lambda e: e[0] == x, encoded_class))[0][1] for x in expected ])

# Encode all the dataset classes
def encode_class(ds):
    return one_hot_encoding(pd.unique(ds.iloc[:,-1:].values.flatten()))

# Decode the result
def decode_result(encoded_class, value):
    if sum(value) != 1:
        value = list(encoded_class[randint(0,len(encoded_class)-1)][1])
    return list(filter(lambda x: list(x[1]) == value,encoded_class))[0][0]
# Class to represent the hidden layer
class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * np.random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1
# Classe to represent the neural network (by default, it was made with two layers)

class NeuralNetwork():
    
    # Construct method
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2
        
    # Sigmoid function
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # sigmoid's derivative
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    # Add bias
    def __add_bias(self, inputs_training):
        return np.array([ np.append(i,-1) for i in inputs_training])
    
    # Train network - Used: Delta Rule
    def train(self, inputs_training, outputs_training, num_interation, rate):
        
        inputs_training = self.__add_bias(inputs_training)
        
        for interate in range(0,num_interation):
            # Calcule the result of neuron
            output_layer1, output_layer2 = self.__think(inputs_training)
            
            # Calcule the layer 2 error
            layer2_error = outputs_training - output_layer2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_layer2)
            
            # Calcule layer 1 error
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_layer1)
            
            # How much adjustment it will take in synaptic weights
            layer1_adjustment = inputs_training.T.dot(layer1_delta) * rate
            layer2_adjustment = output_layer1.T.dot(layer2_delta) * rate
            
            #Adjust synaptic weights
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment
            
    # Return the output of neuron layer
    def __think(self, input_training):
        output_layer1 = self.__sigmoid(np.dot(input_training, self.layer1.synaptic_weights))
        output_layer2 = self.__sigmoid(np.dot(output_layer1, self.layer2.synaptic_weights))
        
        return output_layer1, output_layer2
    
    # Predict passing the datas
    def predict(self, input_):
        input_ = input_ + [-1] 
        h, out = self.__think(input_)
        result = [1 if o >= 0.5 else 0 for o in out]
        return result
        
    # Print weights
    def print_weights(self):
        
        print('Layer 1:')
        print(self.layer1.synaptic_weights)
        print('Layer 2:')
        print(self.layer2.synaptic_weights)
def train(dataset_train, dataset_test):
    count_correct = 0
    count_incorrect = 0
    
    count_by_classes_correct = [0 for i in range(0,len(classes))]
    count_by_classes_incorrect = [0 for i in range(0,len(classes))]
    
    # Encode the classes of dataset
    encoded_class = encode_class(dataset_train)
    
    # Neural network with 2 layers, One with 16 neurons and other with 6 neurons.
    
    # Layer 1: Make 16 neurons com 33 inputs (quantity of input from dataset)
    l1 = NeuronLayer(32,len_cols + 1)
    # Layer 2: Make 12 nerons with 16 input from the other neuron layers (output layer)
    l2 = NeuronLayer(len(classes), 32)
    
    neural_network = NeuralNetwork(l1, l2)
    
    inputs = dataset_train.iloc[:,:-1].values
    outputs = dataset_train.iloc[:,-1:].values
    
    outputs_encoded = encode_expected(outputs,encoded_class)
    
    neural_network.train(inputs, outputs_encoded, 10000, 0.01)
    
    for index, row in dataset_test.iterrows():
        
        tuple_t = list(row)
        tuple_t.pop()
        
        r = neural_network.predict(tuple_t) # Performs the result by the value of the network
        
        result = decode_result(encoded_class, r)
        
        #Result
        if result == row[last_col]:
            count_correct += 1
            count_by_classes_correct[classes.index(result)] += 1
        else:
            count_incorrect += 1
            count_by_classes_incorrect[classes.index(result)] += 1
        
    return count_correct, count_incorrect, count_by_classes_correct, count_by_classes_incorrect
def seperate_ds_by_class(dataset, percentage):
    rds_train = pd.DataFrame()
    rds_test = pd.DataFrame()
    
    for c in classes:
        nds = dataset[dataset[last_col]==c]
        
        ds_train = nds.sample(frac=percentage, random_state=randint(0,15100))
        ds_test = nds.drop(ds_train.index) 
        
        rds_train = rds_train.append(ds_train)
        rds_test = rds_test.append(ds_test)
        
    rds_train = rds_train.reset_index()
    rds_test = rds_test.reset_index() 

    rds_train.drop('index',1,inplace=True) 
    rds_test.drop('index',1,inplace=True) 
    
    return (rds_train, rds_test)
def run_nth(ds,percentage, number):
    percentages_correct = list()
    prob_correct_by_class = []
    
    for i in range(0,number):
        ds_train, ds_test = seperate_ds_by_class(ds,percentage)
        correct, incorrect, count_by_classes_correct, count_by_classes_incorrect = train(ds_train, ds_test)

        by_class = []
        for count_correct, count_incorrect in zip(count_by_classes_correct, count_by_classes_incorrect):
            if count_correct+count_incorrect != 0:
                by_class.append(count_correct/(count_correct+count_incorrect))
            else:
                by_class.append(0)
                
        prob_correct_by_class.append(by_class)
        percentages_correct.append(correct/(correct+incorrect))
        
    return (percentages_correct, prob_correct_by_class)
percents, prob_by_class = run_nth(dss,0.8,1)

taxa_acerto_min=np.min(percents)
taxa_acerto_max=np.max(percents)
taxa_acerto_med=np.mean(percents)

print('Rate')
print('--------------')
print('Min: ' + str(taxa_acerto_min))
print('Max: ' + str(taxa_acerto_max))
print('Mean: '+str(taxa_acerto_med))

print('-------------------------------')
print('Mean rate by class')
print('-------------------------------')

ar_value = [ np.mean(m) for m in np.array(prob_by_class).transpose() ]

for i, _class in enumerate(ar_value):
    print('Class \'' +  str(classes[i]) +'\' : ' + str(_class))





