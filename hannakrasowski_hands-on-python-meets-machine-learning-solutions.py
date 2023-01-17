print('Hello World!')
# Python program to declare variables and data structures (# starts a comment in Python)

print('First, different variables are declared:')

myNumber = 3

print(myNumber) 



myNumber2 = 4.5

print(myNumber2) 



myNumber ="helloworld"

print(myNumber) 



print('Then, different data structures:')

myList = [1,2,3,4]

print(myList)



myDict = {'dog name': 'Bello', 'dog age':3}

print(myDict)
# Python program to illustrate selection statement 



num1 = 34

num2 = 12

if(num1>num2): 

    print("num1 is greater") 

elif(num1<num2): 

    print("num1 is smaller") 

else: 

    print("num1 and num2 are equal") 

# Python program to illustrate how functions are defined



def adder(arg1, arg2): # here the name of the function and the arguments are defined

    result = arg1 + arg2 # this is the body of the function. In this case a simple addition.

    return result # after return the outputs are defined



print('Testing adder with 1 and 5 results in:')

result = adder(2,5) # here we apply the function for the argumens 2 and 5 and store the output in the variable result

print(result) # print the result so we can see what it actually is



print('Testing adder with 1 and -3.5 results in:')

print(adder(1,-3.5)) # another test for the function but we print the output directly

    
# Tip: operators for comparison in Python: <, >, =<, >=, != (not equal)

# CODE HERE:

def substract_positive(num1, num2):

    if num1>num2:

        result = num1-num2

    else:

        result = num2-num1

    return result
print(substract_positive(5,10))

print(substract_positive(-3,4.3))

print(substract_positive(2,2))
# importing packages

import numpy as np # this translates to: import package numpy and store it for usage under the name np

import matplotlib.pyplot as plt
def rel_error(x, y):

    """ returns relative error """

    absolute_difference = np.abs(x - y) # np.abs() returns the absolute value so for -2 it returns 2

    real_value = np.maximum(1e-8, np.abs(x)) # np.maximum(arg1,arg2) returns the greater argument

    

    return absolute_difference / real_value



print(rel_error(2,2.5))
class Min_Neural_Net:

    def __init__(self):

            np.random.seed(10) # for generating the same results

            self.W_1   = np.random.rand(3,4) # input to hidden layer weights

            self.W_2   = np.random.rand(4,1) # hidden layer to output weights

        
def sigmoid(x, w):

    z = np.dot(x, w)

    return 1/(1 + np.exp(-z))

    

def sigmoid_derivative(x, w):

    return sigmoid(x, w) * (1 - sigmoid(x, w))



plt.plot(np.linspace(-10,10,100),sigmoid(np.eye(100),np.linspace(-10,10,100)))

plt.title('Sigmoid function')

plt.show()
# Tip: this formula describes how the prediction is caluclated: y_predict = a(W_2* a(W_1 * x))

# Tip: the weights are stored in the neural network object NN

# Tip: you can access the weight W_1 by writing NN.W_1



def forward_pass(NN, x, y_true):

    loss = None

    x_hidden = None

    

    # CODE HERE:

    x_hidden = sigmoid(x, NN.W_1)

    y_predict = sigmoid(x_hidden, NN.W_2)

    loss = y_true - y_predict

    

    return loss, x_hidden
def backward_pass(NN, loss, x_hidden, x):

    learning_rate = 0.1

    

    # gradients for hidden to output weights

    g_w_2 = np.dot(x_hidden.T, - loss * sigmoid_derivative(x_hidden, NN.W_2))

    # gradients for input to hidden weights

    g_w_1 = np.dot(x.T, np.dot(- loss * sigmoid_derivative(x_hidden, NN.W_2), NN.W_2.T) * sigmoid_derivative(x, NN.W_1))

    

    # update weights

    NN.W_1 = NN.W_1 - learning_rate * g_w_1

    NN.W_2 = NN.W_2 - learning_rate * g_w_2



    return NN
# Initializations

neural_network = Min_Neural_Net()

print('Random starting input to hidden weights: ')

print(neural_network.W_1)

print('Random starting hidden to output weights: ')

print(neural_network.W_2)



# Some data to train the network on

X = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

y = np.array([[0, 1, 1, 0]]).T



# Defining how many updates should be conducted

iterations = 1000

log_loss = []



for i in range(iterations):

    loss, x_hidden = forward_pass(neural_network, X, y)

    neural_network = backward_pass(neural_network, loss, x_hidden, X)

    log_loss.append(np.abs(np.sum(loss)))

    



plt.plot( np.linspace(0,iterations, iterations),log_loss, label='train')

plt.title('Training loss history')

plt.xlabel('Iteration')

plt.ylabel('Loss')



plt.tight_layout()

plt.show()

    

    
# Tip: the prediction is almost the same as the forward pass



def prediction(NN, x):

    y_predict = None

    

    # CODE HERE:

    x_hidden = sigmoid(x, NN.W_1)

    y_predict = sigmoid(x_hidden, NN.W_2)



    return y_predict
print('The final prediction after training for the data is:')

print(prediction(neural_network, X))

print('The absolute error between true values and prediction is:')

print(y-prediction(neural_network, X))    
# A bit of setup



import numpy as np

import matplotlib.pyplot as plt



from neural_net import TwoLayerNet



%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'



# for auto-reloading external modules

# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

%load_ext autoreload

%autoreload 2
from data_utils import load_CIFAR10

from vis_utils import visualize_cifar10



def get_CIFAR10_data(num_training=17000, num_validation=1000, num_test=1000, num_dev=500):

    """

    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare

    it for the linear classifier. 

    """

    # Load the raw CIFAR-10 data

    cifar10_dir = '../input/cifar10/'

    X, y = load_CIFAR10(cifar10_dir)

    

    # Our training set will be the first num_train points from the original

    # training set.

    mask = range(num_training)

    X_train = X[mask]

    y_train = y[mask]

    

    

    # Our validation set will be num_validation points from the original

    # training set.

    mask = range(num_training, num_training + num_validation)

    X_val = X[mask]

    y_val = y[mask]

    

    # We use a small subset of the training set as our test set.

    mask = range(num_training + num_validation, num_training + num_validation + num_test)

    X_test = X[mask]

    y_test = y[mask]

    

    # We will also make a development set, which is a small subset of

    # the training set. This way the development cycle is faster.

    mask = np.random.choice(num_training, num_dev, replace=False)

    X_dev = X_train[mask]

    y_dev = y_train[mask]



    # Normalize the data: subtract the mean image

    mean_image = np.mean(X_train, axis = 0)

    X_train -= mean_image

    X_val -= mean_image

    X_test -= mean_image

    X_dev -= mean_image

    

    # Preprocessing: reshape the image data into rows

    X_train = np.reshape(X_train, (X_train.shape[0], -1))

    X_val = np.reshape(X_val, (X_val.shape[0], -1))

    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))



    return X, y, X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev





# Invoke the above function to get our data.

X_raw, y_raw, X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev= get_CIFAR10_data()

print('Train data shape: {}'.format(X_train.shape))

print('Train labels shape:{}'.format(y_train.shape))

print('Validation data shape: {}'.format(X_val.shape))

print('Validation labels shape:{}'.format(y_val.shape))

print('Test data shape: {}'.format(X_test.shape))

print('Test labels shape: {}'.format(y_test.shape))

print('dev data shape: {}'.format(X_dev.shape))

print('dev labels shape: {}'.format(y_dev.shape))



# visualize raw data

visualize_cifar10(X_raw, y_raw)
# Create a neural network for the task

input_size = 32 * 32 * 3

hidden_size = 50

num_classes = 10

net = TwoLayerNet(input_size, hidden_size, num_classes)
# Train the network

stats = net.train(X_train, y_train, X_val, y_val,

            num_iters=1000, batch_size=200,

            learning_rate=1e-4, learning_rate_decay=0.95,

            reg=0.5, verbose=True)



# Predict on the validation set

val_acc = (net.predict(X_val) == y_val).mean()

print('Validation accuracy: {}'.format(val_acc))
from vis_utils import visualize_grid



# Plot the loss function and train / validation accuracies

plt.subplots(nrows=2, ncols=1)



plt.subplot(2, 1, 1)

plt.plot(stats['loss_history'])

plt.title('Loss history')

plt.xlabel('Iteration')

plt.ylabel('Loss')



plt.subplot(2, 1, 2)

plt.plot(stats['train_acc_history'], label='train')

plt.plot(stats['val_acc_history'], label='val')

plt.title('Classification accuracy history')

plt.xlabel('Epoch')

plt.ylabel('Clasification accuracy')



plt.tight_layout()

plt.show()



# Visualize the weights of the network



def show_net_weights(net):

    fig = plt.figure(figsize=(20,20))

    W1 = net.params['W1']

    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)

    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))

    plt.gca().axis('off')

    plt.show()



show_net_weights(net)
from neural_net import TwoLayerNet, neuralnetwork_hyperparameter_tuning

from model_savers import save_two_layer_net

from data_utils import get_CIFAR10_data_full



### COMMENT IN THE FOLLOWING LINES IF YOU WANT TO TUNE YOUR SELF ON A LARGER DATASET - ONLY POSSIBLE IF LOGGED IN TO KAGGLE DUE TO RESOURCE CONSTRAINTS



# X_raw, y_raw, X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev= get_CIFAR10_data_full() # load three times as large dataset as used before



# best_net, all_classifiers = neuralnetwork_hyperparameter_tuning(X_train, y_train, X_val, y_val) # this is the tuning part. If you want to change the hyperparameters change the function in the neural net util.



# show_net_weights(best_net) # visualize the weights of the best network
from model_savers import load_two_layer_net



net_stored = load_two_layer_net()

show_net_weights(net_stored)
test_acc = (net_stored.predict(X_test) == y_test).mean() 

# test_acc = (best_net.predict(X_test) == y_test).mean() # comment in this line and out the line above if you want to test the model you tuned



print('Test accuracy: ', test_acc)