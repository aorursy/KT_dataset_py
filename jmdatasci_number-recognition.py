import numpy as np

import pandas as pd

import random
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

predicted_results = pd.read_csv('../input/sample_submission.csv')
train_X = train.drop('label', axis=1)/255

train_y = train['label']



samples = test/255



samples.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_X, train_y, test_size=.20, random_state=1)

print("number of test samples :", x_test.shape[0])

print("number of training samples:",x_train.shape[0])
def vectorized_result(j):

    e = np.zeros((10, 1))

    e[j] = 1.0

    return e
training_inputs = []   

test_inputs = []

sample_inputs = []



for row in x_train.iterrows():

    training_inputs.append((row[1].values.reshape((784,1))))

                        

training_results = [vectorized_result(y) for y in y_train]

training_data = list(zip(training_inputs, training_results))

    

for row in x_test.iterrows():

    test_inputs.append((row[1].values.reshape((784,1)))) 



test_data = list(zip(test_inputs, y_test))



for row in samples.iterrows():

    sample_inputs.append((row[1].values.reshape((784,1))))
class Network(object): #constructing network

    def __init__(self,sizes): # sizes is an array of number of neurons per layer

        self.num_layers = len(sizes) #number of hidden layers based on size array chosen

        self.sizes = sizes

        self.biases = [np.random.randn(y,1) for y in sizes[1:]] #randomly generate initial biases

        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])] #randomly generate initial weights



    def sigmoid(z): #build the sigmoid function for output transformation

        return 1.0/(1.0+np.exp(-z))



    def feedforward(self, a):

        for b, w in zip(self.biases, self.weights):

            a = sigmoid(np.dot(w, a)+b) #find the updated 'a' value for each, dot multiply vectors and add biases

        return a



    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):

    # Train the neural network using mini-batch stochastic gradient descent.

    # The "training_data" is a list of tuples (x, y) representing the training inputs and the desired

    # epochs is the number of 'learning loops' the program runs before accepting values

    # eta is the learning rate

        if test_data:

            n_test = sum(1 for _ in test_data) # this is the test data used for scoring after the training is done

        n = sum(1 for _ in training_data)  # training

        for j in range(epochs):  #changed from xrange to range, xrange is depreciated

            random.shuffle(training_data)

            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] #Uses randomly selected mini batches instead of entire dataset

            for mini_batch in mini_batches:

                self.update_mini_batch(mini_batch, eta) # local gradient descent for each mini batch

            if test_data:

                print ("Epoch {}: {} / {} --> {:2f}%".format(j, self.evaluate(test_data), n_test,(100*self.evaluate(test_data)/n_test)))

            else:

                print ("Epoch {} complete".format(j))



    def update_mini_batch(self, mini_batch, eta):

        nabla_b = [np.zeros(b.shape) for b in self.biases] #every bias gets a np array of 0s entire size of biases 

        nabla_w = [np.zeros(w.shape) for w in self.weights] #every weight gets a np array of 0s the entire size of weights

        for x, y in mini_batch: #update the values in the network through backprop algorithm method

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)] #new weights

        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)] #new biases



    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases] # layer by layer vector of biases

        nabla_w = [np.zeros(w.shape) for w in self.weights]# layer by layer vector of weights

    # feedforward

        activation = x

        activations = [x] # list to store all the activations, layer by layer

        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):

            z = np.dot(w, activation)+b

            zs.append(z)

            activation = sigmoid(z)

            activations.append(activation)

        # backward pass

        delta = (activations[-1]-y) * sigmoid_prime(zs[-1])

        nabla_b[-1] = delta

        nabla_w[-1] = np.dot(delta, activations[-2].transpose())



        #l is negative index of list since it is backpropagation

        for l in range(2, self.num_layers):

            z = zs[-l]

            sp = sigmoid_prime(z)

            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            nabla_b[-l] = delta

            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)



    def evaluate(self, test_data):

        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data] #how does generated test result match up with actual

        return sum(int(x == y) for (x, y) in test_results)

    

    def predict(self, samples):

        predicts = [(np.argmax(self.feedforward(x))) for x in samples] #generate results

        return predicts

    

def sigmoid(z): #build the sigmoid function for output transformation

        return 1.0/(1.0+np.exp(-z))

    

def sigmoid_prime(z):

    #Derivative of the sigmoid function

    return sigmoid(z)*(1-sigmoid(z))
mindbrain = Network([784,64,64,10]) # 64 nodes per hidden layer
mindbrain.SGD(training_data, 30, 7, 1.2, test_data=test_data) # each number is 784 pixels, 7 datapoint mini batches, 30 epochs, 1.2 learning rate
results = mindbrain.predict(sample_inputs)
print('Results, frequency per number: \n')

print('0:', results.count(0))

print('1:', results.count(1))

print('2:', results.count(2))

print('3:', results.count(3))

print('4:', results.count(4))

print('5:', results.count(5))

print('6:', results.count(6))

print('7:', results.count(7))

print('8:', results.count(8))

print('9:', results.count(9))
predicted_results.Label = results
predicted_results.head(20)
predicted_results.to_csv('submission.csv', index=False)