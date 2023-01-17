# Numpy handles matrix multiplication, see http://www.numpy.org/

import numpy as np

# PyPlot is a matlab like plotting framework, see https://matplotlib.org/api/pyplot_api.html

import matplotlib.pyplot as plt

# This line makes it easier to plot PyPlot graphs in Jupyter Notebooks

%matplotlib inline
import sklearn

import sklearn.datasets

import matplotlib

# Slightly larger plot rendering

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
class LogisticRegressor:

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

    def __init__(self,input_dim, learning_rate):

        '''

        Assigns the given hyper parameters and initializes the initial parameters.

        '''

        # Assign input dimensionality 

        self.input_dim = input_dim

        # Assign learning rate

        self.learning_rate = learning_rate

        # Trigger parameter setup

        self.init_parameters()

    

    # Parameter setup function

    def init_parameters(self):

        '''

        Initializes weights with random number between -1 and 1

        Initializes bias with 0

        Assigns weights and parameters to model

        '''

        # Randomly init weights

        W1 = 2*np.random.random((self.input_dim,1)) - 1

        # Set bias to 0

        b1 = 0

        # Assign to model

        self.model = {'W1':W1,'b1':b1}

        return

    

    

    # Sigmoid function

    def sigmoid(self,x):

        '''

        Calculates the sigmoid activation of a given input x

        See: https://en.wikipedia.org/wiki/Sigmoid_function

        '''

        return 1/(1+np.exp(-x))



    #Log Loss function

    def log_loss(self,y,y_hat):

        '''

        Calculates the logistic loss between a prediction y_hat and the labels y

        See: http://wiki.fast.ai/index.php/Log_Loss

        

        We need to clip values that get too close to zero to avoid zeroing out. 

        Zeroing out is when a number gets so small that the computer replaces it with 0.

        Therefore, we clip numbers to a minimum value.

        '''

        

        minval = 0.000000000001

        m = y.shape[0]

        l = -1/m * np.sum(y * np.log(y_hat.clip(min=minval)) + (1-y) * np.log((1-y_hat).clip(min=minval)))

        return l



    # Derivative of log loss function

    def log_loss_derivative(self,y,y_hat):

        '''

        Calculates the gradient (derivative) of the log loss between point y and y_hat

        See: https://stats.stackexchange.com/questions/219241/gradient-for-logistic-loss-function

        '''

        return (y_hat-y)



    # Forward prop (forward pass) function    

    def forward_propagation(self,A0):

        '''

        Forward propagates through the model, stores results in cache.

        See: https://stats.stackexchange.com/questions/147954/neural-network-forward-propagation

        A0 is the activation at layer zero, it is the same as X

        '''

        # Load parameters from model

        W1, b1 = self.model['W1'],self.model['b1']

        # Do the linear step

        z1 = A0.dot(W1) + b1



        #Pass the linear step through the activation function

        A1 = self.sigmoid(z1)

        # Store results in cache

        self.cache = {'A0':X,'z1':z1,'A1':A1}

        return

    

    # Backprop function

    def backward_propagation(self,y):

        '''

        Backward propagates through the model to calculate gradients.

        Stores gradients in grads dictionary.

        See: https://en.wikipedia.org/wiki/Backpropagation

        '''

        # Load results from forward pass

        A0, z1, A1 = self.cache['A0'],self.cache['z1'], self.cache['A1']

        # Load model parameters

        W1, b1 = self.model['W1'], self.model['b1']

        

        # Read m, the number of examples

        m = A0.shape[0]

        # Calculate the gradient of the loss function

        dz1 = self.log_loss_derivative(y=y,y_hat=A1)

        # Calculate the derivative of the loss with respect to the weights W1

        dW1 = 1/m*(A0.T).dot(dz1)

        # Calculate the derivative of the loss with respect to the bias b1

        db1 = 1/m*np.sum(dz1, axis=0, keepdims=True)

        

        #Make sure the weight derivative has the same shape as the weights

        assert(dW1.shape == W1.shape)

        

        # Store gradients in gradient dictionary

        self.grads = {'dW1':dW1,'db1':db1}

        return

    

    # Parameter update

    def update_parameters(self):

        '''

        Updates parameters accoarding to gradient descent algorithm

        See: https://en.wikipedia.org/wiki/Gradient_descent

        '''

        # Load model parameters

        W1, b1 = self.model['W1'],self.model['b1']

        # Load gradients

        dW1, db1 = self.grads['dW1'], self.grads['db1']

        # Update weights

        W1 -= self.learning_rate * dW1

        # Update bias

        b1 -= self.learning_rate * db1

        # Store new parameters in model dictionary

        self.model = {'W1':W1,'b1':b1}

        return

    

    # Prediction function

    def predict(self,X):

        '''

        Predicts y_hat as 1 or 0 for a given input X

        '''

        # Do forward pass

        self.forward_propagation(X)

        # Get output of regressor

        regressor_output = self.cache['A1']

        

        # Turn values to either 1 or 0

        regressor_output[regressor_output > 0.5] = 1

        regressor_output[regressor_output < 0.5] = 0

        

        # Return output

        return regressor_output

    # Train function

    def train(self,X,y, epochs):

        '''

        Trains the regressor on a given training set X, y for the specified number of epochs.

        '''

        # Set up array to store losses

        losses = []

        # Loop through epochs

        for i in range(epochs):

            # Forward pass

            self.forward_propagation(X)

            

            # Calculate loss

            loss = self.log_loss(y,self.cache['A1'])

            # Store loss

            losses.append(loss)

            # Print loss every 10th iteration

            if (i%10 == 0):

                print('Epoch:',i,' Loss:', loss)

            

            # Do the backward propagation

            self.backward_propagation(y)

            # Update parameters

            self.update_parameters()

        # Return losses for analysis

        return losses
#Seed the random function to ensure that we always get the same result

np.random.seed(1)



#Variable definition



#define X

X = np.array([[0,1,0],

              [1,0,0],

              [1,1,1],

              [0,1,1]])

#define y

y = np.array([[0,1,1,0]]).T



# Define instance of class

regressor = LogisticRegressor(input_dim=3,learning_rate=1)
# Train classifier

losses = regressor.train(X,y,epochs=100)
# Plot the losses for analyis

plt.plot(losses)
# Generate a dataset and plot it

np.random.seed(0)

X, y = sklearn.datasets.make_blobs(n_samples=200,centers=2)

y = y.reshape(200,1)

plt.scatter(X[:,0], X[:,1], s=40, c=y.flatten(), cmap=plt.cm.Spectral)
# Define instance of class

# Learning rate = 1, same as no learning rate used

regressor = LogisticRegressor(input_dim=2,learning_rate=10)
# Train classifier



losses = regressor.train(X,y,epochs=100)
plt.plot(losses)
# Define instance of class

# Learning rate = 0.05

regressor = LogisticRegressor(input_dim=2,learning_rate=0.05)
# Train classifier

losses = regressor.train(X,y,epochs=100)
plt.plot(losses)
# Define instance of class

# Learning rate = 0.0005

regressor = LogisticRegressor(input_dim=2,learning_rate=0.0005)
# Train classifier

losses = regressor.train(X,y,epochs=100)
plt.plot(losses)
# Define instance of class

# Tweak learning rate here

regressor = LogisticRegressor(input_dim=2,learning_rate=1)
# Train classifier

losses = regressor.train(X,y,epochs=100)
plt.plot(losses)
# Helper function to plot a decision boundary.

# If you don't fully understand this function don't worry, it just generates the boundary plot.

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

    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.Spectral)
# Define instance of class

# Learning rate = 0.05

regressor = LogisticRegressor(input_dim=2,learning_rate=0.05)

# Train classifier

losses = regressor.train(X,y,epochs=100)
# Plot the decision boundary

plot_decision_boundary(lambda x: regressor.predict(x))

plt.title("Decision Boundary for logistic regressor")
# Generate a dataset and plot it

np.random.seed(0)

X, y = sklearn.datasets.make_moons(200, noise=0.1)

y = y.reshape(200,1)

plt.scatter(X[:,0], X[:,1], s=40, c=y.flatten(), cmap=plt.cm.Spectral)
# Define instance of class

# Learning rate = 0.05

y = y.reshape(200,1)

regressor = LogisticRegressor(input_dim=2,learning_rate=0.05)

# Train classifier

losses = regressor.train(X,y,epochs=100)
# Plot the decision boundary

plot_decision_boundary(lambda x: regressor.predict(x))

plt.title("Decision Boundary for hidden layer size 3")