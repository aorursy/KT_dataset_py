#importing the required libraries
from random import seed
import numpy as np
#function to create a network
#num_inputs = number of inputs to the network
#num_hidden_layers = number of hidden layers
#num_nodes_hidden = array of number of nods in each layer
#num_nodes_output = number of output nodes

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    
    num_nodes_previous = num_inputs # number of nodes in the previous layer

    network = {} #initialize an empty network
    
    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer in range(num_hidden_layers + 1):
        
        if layer == num_hidden_layers:
            layer_name = 'output' # name last layer in the network output
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1) # otherwise give the layer a number
            num_nodes = num_nodes_hidden[layer] 
        
        # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
    
        num_nodes_previous = num_nodes

    return network # return the network
#function to view the network
def view_network(network):
    for layer in network:
        print(layer)
        for node in network[layer]:
            print('    ', end='')
            print('{} : {}'.format(node, network[layer][node]))
#function to compute the sum at each node => z = w.x + b
def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias
#function to calculate the value after activation by sigmoid function
#=> a = 1 / (1 + exp(-z))
def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))
#function to propogate forward in the network till the output layer
def forward_propagate(network, inputs):
    
    layer_inputs = list(inputs) # start with the input layer as the input to the first hidden layer
    
    for layer in network:
        
        layer_data = network[layer]
        
        layer_outputs = [] 
        for layer_node in layer_data:
        
            node_data = layer_data[layer_node]
        
            # compute the weighted sum and the output of each node at the same time 
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))
            
        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))
    
        layer_inputs = layer_outputs # set the output of this layer to be the input to next layer

    network_predictions = layer_outputs
    return network_predictions
#creating an example network
#input format : (num_inputs, num_hidden_layers, num_nodes_hidden_in_each_layer, num_nodes_output)
my_network = initialize_network(5, 3, [2, 3, 2], 3)
#viewing our network
view_network(my_network)
#initializing random inputs
inputs = np.around(np.random.uniform(size=5), decimals=2)
#making predictions after passing the network and inputs to the forward_prapogate function.
predictions = forward_propagate(my_network, inputs)
print('The predicted values by the network for the given input are {}'.format(predictions))