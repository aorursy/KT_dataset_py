## I have accepted the challenge and wrote the code based on chapter 3 of Marsland's book 

## Machine Learning - An Algorithmic Perspective where the author describes how to write a perceptron

## The code below has 4 classical tests for perceptron - OR, AND, NOT and XOR



## HOW TO RUN: just run the code, go below and (un)comment the desired tests



import numpy as np



class perceptron:

    def __init__(self, input_list, target_list, eta=0.25, iter=5, add_bias=True):

        self.inputs = input_list

        self.targets = target_list

        self.eta = eta

        self.iter = iter

        

        if add_bias:

            self.matrix = self.add_bias_node(input_list)

        else:

            self.matrix = self.inputs

        

        no_of_col = self.matrix.shape[1] #number of columns plus 1 for bias

        self.weights =np.random.rand(no_of_col, 1)*0.1-0.05

        

    #add a bias node at the beginning of the inputs (bias node has values -1)

    def add_bias_node(self, inputs):

        bias_node = -np.ones((inputs.shape[0],1))

        return np.concatenate((bias_node, inputs), axis=1)

 

    #function calculates activations = input nodes x weights (X x W), checks the thresholds and adjusts the weights

    def iteration(self):

        activations = np.dot(self.matrix, self.weights)

        

        self.weights -= self.eta * np.dot(np.transpose(self.matrix), activations-self.targets)

        

        #thresholds

        thresholds = np.where(activations>0.5, 1, 0)

        print("Thresholds:\n %s" % (thresholds,))



    #train the perceptron with n iterations    

    def train(self):

        print (self.inputs)

        for i in range(1, self.iter+1):

            print("Iteration: %i" % (i,))

            #print("x:%d, prime YES" % (x,))

            print("Weights:\n %s" % (self.weights,))

            self.iteration()



##OR LOGIC

def or_logic():            

    inputs = np.array([[0,0], [0,1], [1,0], [1,1]])

    targets = np.array([[0], [1], [1], [1]])

    or_logic = perceptron(inputs, targets)

    or_logic.train()    

    

##AND LOGIC

def and_logic():

    inputs = np.array([[0,0], [0,1], [1,0], [1,1]])

    targets = np.array([[0], [0], [0], [1]])

    and_logic = perceptron(inputs, targets)

    and_logic.train()



##NOT LOGIC

def not_logic():

    inputs = np.array([[1], [0]])

    targets = np.array([[0], [1]])

    not_logic = perceptron(inputs, targets)

    not_logic.train()



#XOR LOGIC 

def xor_logic():

    inputs = np.array([[0,0,1],[0,1,0],[1,0,0],[1,1,0]])

    targets = np.array([[0], [1], [1], [0]])

    not_logic = perceptron(inputs, targets, add_bias=True, iter=15)

    not_logic.train()



####RUN####

or_logic()

#and_logic()

#not_logic()

#xor_logic()
