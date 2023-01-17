# Package imports
# Matplotlib is a matlab like plotting library
import matplotlib
import matplotlib.pyplot as plt
# Numpy handles matrix operations
import numpy as np
#Panda
import pandas as pd
# SciKitLearn is a useful machine learning utilities library
import sklearn
# The sklearn dataset module helps generating datasets
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
# Display plots inline and change default figure size
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates decision boundary plot
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Accent)
df = pd.read_csv('../input/W1data.csv')
df.head()
# Get labels
y = df[['Cultivar 1', 'Cultivar 2', 'Cultivar 3']].values
# Get inputs
X = df.drop(['Cultivar 1', 'Cultivar 2', 'Cultivar 3'], axis = 1)

X.shape, y.shape


class twolayernn: 
    # Here we are just setting up some placeholder variables
    # This is the dimensionality of our input, that is how many features our input has
    input_dim = 0
    # This is the learning rate alpha
    learning_rate = 0.1
    # We will store the parameters of our model in a dictionary
    model = {}
    # The values calculated in the forward propagation will be stored in this dictionary
    cache = {}
    # The gradients that we calculate during back propagation will be stored in a dictionary
    gradients = {}

 # Init function of the class
    def __init__(self, nn_input_dim, nn_hdim, nn_output_dim, learning_rate):
        # Assign input dimensionality 
        self.nn_input_dim = nn_input_dim
        self.nn_output_dim = nn_output_dim
        
        self.nn_hdim = nn_hdim
        # Assign learning rate
        self.learning_rate = learning_rate
        
        # Trigger parameter setup
        self.init_parameters()
        
    def initialize_parameters(nn_input_dim,nn_hdim,nn_output_dim):
        '''
        Initializes weights with random number between -1 and 1
        Initializes bias with 0
        Assigns weights and parameters to model
        '''
    # First layer weights
        W1 = 2 *np.random.randn(self.nn_input_dim, self.nn_hdim) - 1
    
    # First layer bias
        b1 = np.zeros((1, self.nn_hdim))
    
    # Second layer weights
        W2 = 2 * np.random.randn(self.nn_hdim, self.nn_output_dim) - 1
    
    # Second layer bias
        b2 = np.zeros((1, self.nn_output_dim))
    
    # Package and return model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        return model
    
    #Softmax function
    def softmax(self,z):
        '''
        Calculates the softmax activation of a given input x
        See: https://en.wikipedia.org/wiki/Softmax_function
        '''
    #Calculate exponent term first
        exp_scores = np.exp(z)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def softmax_loss(self,y,y_hat):
        '''
        Calculates the generalized logistic loss between a prediction y_hat and the labels y
        See: http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/

        We need to clip values that get too close to zero to avoid zeroing out. 
        Zeroing out is when a number gets so small that the computer replaces it with 0.
        Therefore, we clip numbers to a minimum value.
        '''
    # Clipping calue
        minval = 0.000000000001
    # Number of samples
        m = y.shape[0]
    # Loss formula, note that np.sum sums up the entire matrix and therefore does the job of two sums from the formula
        loss = -1/m * np.sum(y * np.log(y_hat.clip(min=minval)))
        return loss

    def loss_derivative(self,y,y_hat):
        '''
        Calculates the gradient (derivative) of the loss between point y and y_hat
        See: https://stats.stackexchange.com/questions/219241/gradient-for-logistic-loss-function
        '''
        return (y_hat-y)

    def tanh_derivative(self,x):
        '''
        Calculates the derivative of the tanh function that is used as the first activation function
        See: https://socratic.org/questions/what-is-the-derivative-of-tanh-x
        '''
        return (1 - np.power(x, 2))




    def forward_prop(self,model,a0):
        '''
        Forward propagates through the model, stores results in cache.
        See: https://stats.stackexchange.com/questions/147954/neural-network-forward-propagation
        A0 is the activation at layer zero, it is the same as X
        '''
    
    # Load parameters from model
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Linear step
        z1 = a0.dot(W1) + b1
    
    # First activation function
        a1 = np.tanh(z1)
    
    # Second linear step
        z2 = a1.dot(W2) + b2
    
    # Second activation function
        a2 = softmax(z2)
        cache = {'a0':a0, 'z1':z1,'a1':a1, 'z2':z2,'a2':a2}
        return cache
    def backward_prop(self,model,cache,y):
        '''
        Backward propagates through the model to calculate gradients.
        Stores gradients in grads dictionary.
        See: https://en.wikipedia.org/wiki/Backpropagation
        '''
    # Load parameters from model
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Load forward propagation results
        a0,a1, a2 = cache['a0'],cache['a1'],cache['a2']
    
    # Get number of samples
        m = y.shape[0]
    
    # Backpropagation
    # Calculate loss derivative with respect to output
        dz2 = loss_derivative(y=y,y_hat=a2)
   
    # Calculate loss derivative with respect to second layer weights
        dW2 = 1/m*(a1.T).dot(dz2)
    
    # Calculate loss derivative with respect to second layer bias
        db2 = 1/m*np.sum(dz2, axis=0)
    
    # Calculate loss derivative with respect to first layer
        dz1 = np.multiply(dz2.dot(W2.T) ,tanh_derivative(a1))
    
    # Calculate loss derivative with respect to first layer weights
        dW1 = 1/m*np.dot(a0.T, dz1)
    
    # Calculate loss derivative with respect to first layer bias
        db1 = 1/m*np.sum(dz1, axis=0)
    
    # Store gradients
        grads = {'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}
        return grads




    def update_parameters(self,model,grads,learning_rate):
        '''
        Updates parameters accoarding to gradient descent algorithm
        See: https://en.wikipedia.org/wiki/Gradient_descent
        '''
    # Load parameters
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Update parameters
        W1 -= learning_rate * grads['dW1']
        b1 -= learning_rate * grads['db1']
        W2 -= learning_rate * grads['dW2']
        b2 -= learning_rate * grads['db2']
    
    # Store and return parameters
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        return model

    def predict(self,model, x):
        '''
        Predicts y_hat as 1 or 0 for a given input X
        '''
    # Do forward pass
        c = forward_prop(model,x)
    #get y_hat
        y_hat = np.argmax(c['a2'], axis=1)
        return y_hat
    def train(self,model,X_,y_,learning_rate, epochs=20000, print_loss=False):
    # Gradient descent. Loop over epochs
        for i in range(0, epochs):

        # Forward propagation
            cache = forward_prop(model,X_)
        #a1, probs = cache['a1'],cache['a2']
        # Backpropagation
        
            grads = backward_prop(model,cache,y_)
        # Gradient descent parameter update
        # Assign new parameters to the model
            model = update_parameters(model=model,grads=grads,learning_rate=learning_rate)
    
        # Pring loss & accuracy every 100 iterations
            if print_loss==True and i % 100 == 0:
                a2 = cache['a2']
                print('Loss after iteration',i,':',softmax_loss(y_,a2))
                y_hat = predict(model,X_)
                y_true = y_.argmax(axis=1)
                print('Accuracy after iteration',i,':',accuracy_score(y_pred=y_hat,y_true=y_true)*100,'%')
    
        return model

# Hyper parameters
hiden_layer_size = 5
# I picked this value because it showed good results in my experiments
learning_rate = 0.1
# Initialize the parameters to random values. We need to learn these.
np.random.seed(0)
# This is what we return at the end
model = twolayernn.initialize_parameters(nn_input_dim=13, nn_hdim= hiden_layer_size, nn_output_dim= 3)
model = twolayernn.train(model,X,y,learning_rate=learning_rate,epochs=1000,print_loss=True)

