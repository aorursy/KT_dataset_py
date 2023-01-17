from sklearn import datasets     #for dataset

import numpy as np               #for maths

import matplotlib.pyplot as plt  #for plotting
iris = datasets.load_iris()      #load the dataset

data = iris.data                 #get features  

target = iris.target             #get labels



shape = data.shape               #shape of data



#convert into numpy array

data = np.array(data).reshape(shape[0],shape[1])

target = np.array(target).reshape(shape[0],1)



#print shape

print("Data Shape   = {}".format(data.shape))

print("Target Shape = {}".format(target.shape))

print('Classes : {}'.format(np.unique(target)))

print('Sample data : {} , Target = {}'.format(data[70],target[70]))
#HYPERPARAMETERS



#num of target labels

num_classes = len(np.unique(target))



#define layer_neurons

input_units  = 4   #neurons in input layer

hidden_units = 8   #neurons in hidden layer

output_units = 3   #neurons in output layer



#define hyper-parameters

learning_rate = 0.03



#regularization parameter

beta = 0.00001



#num of iterations

iters = 4001
#PARAMETERS



#initialize parameters i.e weights

def initialize_parameters():

    #initial values should have zero mean and 0.1 standard deviation

    mean = 0        #mean of parameters 

    std = 0.03      #standard deviation

    

    layer1_weights = np.random.normal(mean,std,(input_units,hidden_units))          

    layer1_biases = np.ones((hidden_units,1))                                       

    layer2_weights = np.random.normal(mean,std,(hidden_units,output_units))

    layer2_biases = np.ones((output_units,1))

    

    parameters = dict()

    parameters['layer1_weights'] = layer1_weights

    parameters['layer1_biases'] = layer1_biases

    parameters['layer2_weights'] = layer2_weights

    parameters['layer2_biases'] = layer2_biases

    

    return parameters
#activation function

def sigmoid(X):

    return 1/(1+np.exp((-1)*X))



#softmax function for output

def softmax(X):

    exp_X = np.exp(X)

    exp_X_sum = np.sum(exp_X,axis=1).reshape(-1,1)

    exp_X = (exp_X/exp_X_sum)

    return exp_X
#forward propagation

def forward_propagation(train_dataset,parameters):

    cache = dict()            #to store the intermediate values for backward propagation

    m = len(train_dataset)    #number of training examples

    

    #get the parameters

    layer1_weights = parameters['layer1_weights']

    layer1_biases = parameters['layer1_biases']

    layer2_weights = parameters['layer2_weights']

    layer2_biases = parameters['layer2_biases']

    

    #forward prop

    logits = np.matmul(train_dataset,layer1_weights) + layer1_biases.T

    activation1 = np.array(sigmoid(logits)).reshape(m,hidden_units)

    activation2 = np.array(np.matmul(activation1,layer2_weights) + layer2_biases.T).reshape(m,output_units)

    output = np.array(softmax(activation2)).reshape(m,num_classes)

    

    #fill in the cache

    cache['output'] = output

    cache['activation1'] = activation1

    

    return cache,output



#backward propagation

def backward_propagation(train_dataset,train_labels,parameters,cache):

    derivatives = dict()         #to store the derivatives

    

    #get stuff from cache

    output = cache['output']

    activation1 = cache['activation1']

    

    #get parameters

    layer1_weights = parameters['layer1_weights']

    layer2_weights = parameters['layer2_weights']

    

    #calculate errors

    error_output = output - train_labels

    error_activation1 = np.matmul(error_output,layer2_weights.T)

    error_activation1 = np.multiply(error_activation1,activation1)

    error_activation1 = np.multiply(error_activation1,1-activation1)

    

    

    #calculate partial derivatives

    partial_derivatives2 = np.matmul(activation1.T,error_output)/len(train_dataset)

    partial_derivatives1 = np.matmul(train_dataset.T,error_activation1)/len(train_dataset)

    

    #store the derivatives

    derivatives['partial_derivatives1'] = partial_derivatives1

    derivatives['partial_derivatives2'] = partial_derivatives2

    

    return derivatives





#update the parameters

def update_parameters(derivatives,parameters):

    #get the parameters

    layer1_weights = parameters['layer1_weights']

    layer2_weights = parameters['layer2_weights']

    

    #get the derivatives

    partial_derivatives1 = derivatives['partial_derivatives1']

    partial_derivatives2 = derivatives['partial_derivatives2']

    

    #update the derivatives

    layer1_weights -= (learning_rate*(partial_derivatives1 + beta*layer1_weights))

    layer2_weights -= (learning_rate*(partial_derivatives2 + beta*layer2_weights))

    

    #update the dict

    parameters['layer1_weights'] = layer1_weights

    parameters['layer2_weights'] = layer2_weights

    

    return parameters

    

#calculate the loss and accuracy

def cal_loss_accuray(train_labels,predictions,parameters):

    #get the parameters

    layer1_weights = parameters['layer1_weights']

    layer2_weights = parameters['layer2_weights']

    

    #cal loss and accuracy

    loss = -1*np.sum(np.multiply(np.log(predictions),train_labels) + np.multiply(np.log(1-predictions),(1-train_labels)))/len(train_labels) + np.sum(layer1_weights**2)*beta/len(train_labels) + np.sum(layer2_weights**2)*beta/len(train_labels)

    accuracy = np.sum(np.argmax(train_labels,axis=1)==np.argmax(predictions,axis=1))

    accuracy /= len(train_dataset)

    

    return loss,accuracy
#Implementation of 3 layer Neural Network



#training function

def train(train_dataset,train_labels,iters=2):

    #To store loss after every iteration.

    J = []

  

    #WEIGHTS

    global layer1_weights

    global layer1_biases

    global layer2_weights

    global layer2_biases

  

    #initialize the parameters

    parameters = initialize_parameters()

    

    layer1_weights = parameters['layer1_weights']

    layer1_biases = parameters['layer1_biases']

    layer2_weights = parameters['layer2_weights']

    layer2_biases = parameters['layer2_biases']

    

    #to store final predictons after training

    final_output = []

    

    for j in range(iters):

        #forward propagation

        cache,output = forward_propagation(train_dataset,parameters)

        

        #backward propagation

        derivatives = backward_propagation(train_dataset,train_labels,parameters,cache)

        

        #calculate the loss and accuracy

        loss,accuracy = cal_loss_accuray(train_labels,output,parameters)

        

        #update the parameters

        parameters = update_parameters(derivatives,parameters)

        

        #append loss

        J.append(loss)

        

        #update final output

        final_output = output

        

        #print accuracy and loss

        if(j%500==0):

            print("Step %d"%j)

            print("Loss %f"%loss)

            print("Accuracy %f%%"%(accuracy*100))

    

    return J,final_output
#shuffle the dataset

z = list(zip(data,target))

np.random.shuffle(z)

data,target = zip(*z)



#make train_dataset and train_labels

train_dataset = np.array(data).reshape(-1,4)

train_labels = np.zeros([train_dataset.shape[0],num_classes])



#one-hot encoding

for i,label in enumerate(target):

    train_labels[i,label] = 1



#normalizations

for i in range(input_units):

    mean = train_dataset[:,i].mean()

    std = train_dataset[:,i].std()

    train_dataset[:,i] = (train_dataset[:,i]-mean)/std
#train data

J,final_output = train(train_dataset,train_labels,iters=4001)
#plot loss graph

plt.plot(list(range(1,len(J))),J[1:])

plt.xlabel('Iterations')

plt.ylabel('Loss')

plt.title('Iterations VS Loss')

plt.show()