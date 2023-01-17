import numpy as np
def sigmoid(x):

    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):

    return x * (1 - x)
training_inputs = np.array([

                               [0,0,1],

                               [1,1,1],

                               [1,0,1],

                               [0,1,1]

                           ]

                          )

training_outputs = np.array([[0,1,1,0]]).T
np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1)) - 1



print("Random Starting Synaptic Weights : ")

print(synaptic_weights)
for iteration in range(1):

    input_layer = training_inputs

    

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    

print("Outputs after Training : ")

print(outputs)
# Lets modify the same function to calculate the errors and adjust the weights.

for iteration in range(20000): # Play around the iteration to get accurate outputs. Start with 1;5;10;100;1000;10000; 20000

    input_layer = training_inputs

    

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    

    error = training_outputs - outputs

    

    adjustments = error * sigmoid_derivative(outputs)

    

    synaptic_weights += np.dot(input_layer.T, adjustments)

    

print('Synaptic Weights after Training : ')

print(synaptic_weights)

print('-'*20)

print("Outputs after Training : ")

print(outputs)
# import numpy as np
class NeuralNetwork():

    

    def __init__(self):

        np.random.seed(1)

        

        self.synaptic_weights = 2 * np.random.random((3,1)) - 1

        

    def sigmoid(self, x):

        return 1 / (1+np.exp(-x))

    

    def sigmoid_derivative(self, x):

        return x * (1-x)

    

    def train(self, training_inputs, training_outputs, training_iterations):

        for  iteration in range(training_iterations):

            output = self.think(training_inputs)

            error = training_outputs - output

            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

            

    def think(self, inputs):

        inputs = inputs.astype(float)

        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output

    
# if __name__ = "__main__":

neural_network = NeuralNetwork()



print("Random Synaptic Weights")

print(neural_network.synaptic_weights)



neural_network.train(training_inputs, training_outputs, 10000)



print("Random Synaptic Weights After Training")

print(neural_network.synaptic_weights)

# Test this.

test_input = np.array([1,1,1])



print("Output Data for {}: ".format(test_input))

print(neural_network.think(test_input))



print('-*'*20)



test_input = np.array([1,0,0])



print("Output Data for {}: ".format(test_input))

print(neural_network.think(test_input))



print('-*'*20)



test_input = np.array([0,0,1])



print("Output Data for {}: ".format(test_input))

print(neural_network.think(test_input))