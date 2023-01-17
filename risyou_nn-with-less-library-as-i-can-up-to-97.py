import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



#Load Mnist data into a pandas format.

train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')
print('train.csv with', train.shape)

print('test.csv with', test.shape)

train.head(3) # Check the first 3 line of the train.csv
#Seperate the label and image

label_train = train['label']

image_train = train.drop(['label'],axis=1)



# Check the label_train and image_train format. .head(4) returns the first 4 rows.

print(label_train.head(4))

image_train.head(4)
# Change the data to numpy format

label_train_number = np.array(label_train)

image_train = np.array(image_train)



label_train_number
label_train = np.zeros((42000,10))

label_train[np.arange(42000), label_train_number] = 1

label_train
image_test = np.array(test)
choice_number = 1325 # You can choice any number between 0 to 42000.

image_to_plot = image_train[choice_number].reshape((28,28))

label_to_plot = label_train[choice_number]

label_to_plot_number = label_train_number[choice_number]
plt.imshow(image_to_plot)

print('This number is: ', label_to_plot_number)

print('In one-hot format : ', label_to_plot)
def sigmoid(x):

    return 1/(1 + np.exp(-x))



#Gradiation for sigmoid

def sigmoid_grad(x):

    return (1 - sigmoid(x)) * sigmoid(x)



def rule(x):

    return np.maximum(0, x) # Just return x when it is > 0, otherways return 0.



def rule_grad(x):

    grad = np.zeros(x)

    grad[x>0] = 1

    return grad
#Plot two activation.

x = np.arange(-10, 10, 0.2)

markers = {'Sigmoid':'o', 'Relu':'s'}

plt.plot(x, sigmoid(x), label='Sigmoid')

plt.plot(x, rule(x)/10, label='ReLu')

plt.legend(loc='lower right')
def softmax(x):

    if x.ndim == 2:

        x = x.T

        x = x - np.max(x, axis=0)

        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T 



    x = x - np.max(x)

    return np.exp(x) / np.sum(np.exp(x))
x = np.array([0, 0.1, 0.15, 4, 0.4, 0.1, 0.05, 0, 0.1, 0.2]) # Build an array with 10 element.

y = softmax(x)

print(y)

print('The summation is of y: ' ,np.sum(y))
# y - prediction, t - teaching label.

#def cross_entropy_error(y, t):

#    batch_size = y.shape[0] # This makes us able to calculate a batch of data.

#    f = t * np.log(y)

#    return -np.sum(f) / batch_size



def cross_entropy_error(y, t):

    if y.ndim == 1:

        t = t.reshape(1, t.size)

        y = y.reshape(1, y.size)

        

    if t.size == y.size:

        t = t.argmax(axis=1)

             

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
class TwolayerNN:

    

    #Initialize all variables when create a two layer NN.

    def __init__(self, input_size, hidden_size, output_size, wstd = 0.01):

        # Initialize weight and basis.

        # Weights are created by given shape and populate it with random samples from a uniform distributions.

        self.params = {}

        self.params['W1'] = wstd * np.random.randn(input_size, hidden_size)

        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = wstd * np.random.randn(hidden_size, output_size)

        self.params['b2'] = np.zeros(output_size)

    

    #Make prediction. x is 784 pixes, y = 10 number represent the posibility of digits from 0 ~ 9

    def predict(self, x):

        W1 = self.params['W1'] # Recall variables in the class.

        W2 = self.params['W2']

        b1 = self.params['b1']

        b2 = self.params['b2']

        

        a1 = np.dot(x, W1) + b1

        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2

        y = softmax(a2)

        

        return y

    

    #x = input image data, t = teaching label.

    def error(self, x, t):

        y = self.predict(x)

        return cross_entropy_error(y, t)

    

    def accuracy(self, x, t):

        y = self.predict(x)

        y = y.argmax(axis=1)

        t = t.argmax(axis=1)

        # Accuracy = (how many prediction are equal with answer in a batch) / (batch size)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    

    def gradient(self, x, t):

        W1 = self.params['W1'] # Recall variables in the class.

        W2 = self.params['W2']

        b1 = self.params['b1']

        b2 = self.params['b2']

        

        grads = {} #Creat a empty dict for gradient.

        

        batch_size = x.shape[0]

        

        #First, get the current a1, z1 etc..

        a1 = np.dot(x, W1) + b1

        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2

        y = softmax(a2)

        

        #Second, calculate the gradient for each variable.

        dy = (y - t) / batch_size

        grads['W2'] = np.dot(z1.T, dy)

        grads['b2'] = np.sum(dy, axis=0)

        

        da1 = np.dot(dy, W2.T)

        dz1 = sigmoid_grad(a1) * da1

        grads['W1'] = np.dot(x.T, dz1)

        grads['b1'] = np.sum(dz1, axis=0)



        return grads
#Initialize our NN with name 'net'

net = TwolayerNN(784, 100, 10)



#Check our parameter dimention

print(net.params['W1'].shape)

print(net.params['b1'].shape)

print(net.params['W2'].shape)

print(net.params['b2'].shape)
#Using current variable, we input 1325th image and try to predict what it is.

print(net.predict(image_train[1325]))
network = TwolayerNN(784, 100, 10) # Create a two-layer NN named 'network'

data_size = image_train.shape[0] #Get the intire data size

learn_rate = 0.01



#We will train our data with a random batch, here we define the size of each batch.

batch_size = 200

#How many times we are going to train our network.

iter_num = 10000



#Create a empty list to store our error

train_error_list = []

train_accuracy_list = []



for i in range(iter_num):

    #

    batch = np.random.choice(data_size, batch_size)

    image_batch = image_train[batch]

    label_batch = label_train[batch]



    grad = network.gradient(image_batch, label_batch)

    

    for key in grad:

        network.params[key] -= grad[key]*learn_rate



    

    error = network.error(image_batch, label_batch)

    train_error_list.append(error)



    if i % 500 == 0:

        accuracy = network.accuracy(image_train, label_train)

        train_accuracy_list.append(accuracy)

        print('Train error:' + str(error) + '  accuracy :' + str(accuracy))
plt.plot(train_error_list)
plt.plot(train_accuracy_list)
label_test = network.predict(image_test) # Predict label with our mode.

label_test = label_test.argmax(axis=1) #Transfor the answer from one-hot format to label.
index = np.arange(1,label_test.shape[0]+1) #Create a index columns

test_predict = pd.DataFrame([index, label_test]) # Store our prediction with index.

test_predict = test_predict.T #transpose our data from 2 line to 2 columns.

test_predict.columns = ['ImageId', 'Label']

test_predict.head(5)
test_predict.to_csv('test_predict', index = False)