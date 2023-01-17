# Display figure in the notebook

%matplotlib inline
# Load packages

import random



import keras

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from sklearn import preprocessing

from sklearn.model_selection import train_test_split



from keras import initializers, optimizers

from keras.layers.core import Dense, Activation

from keras.models import Sequential

from keras.utils.np_utils import to_categorical
# Define some functions

def plot_mnist(data, index, label=None):

    """Plot one image from the mnist dataset."""

    fig = plt.figure(figsize=(3, 3))

    if type(data) == pd.DataFrame:

        plt.imshow(np.asarray(data.iloc[index, 1:]).reshape((HEIGHT, WIDTH)),

                   cmap=plt.cm.gray_r,

                   interpolation='nearest')

        plt.title(f"Image label: {data.loc[index, 'label']}")

    else:

        plt.imshow(data[index].reshape((HEIGHT, WIDTH)),

                   cmap=plt.cm.gray_r,

                   interpolation='nearest')

        plt.title(f"Image label: {label}")

    

    plt.axis('off')

    return fig



def plot_prediction(X, y, model, sample_idx=0, classes=range(10), model_from=None):

    """Plot the prediction for a given image."""

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))



    # Plot the image

    ax0.imshow(scaler.inverse_transform(X[sample_idx]).reshape(HEIGHT, WIDTH), 

               cmap=plt.cm.gray_r,

               interpolation='nearest')

    ax0.axis('off')

    ax0.set_title(f"True image label: {y[sample_idx]}");



    # Plot the predictions

    ax1.bar(classes, one_hot(len(classes), y[sample_idx]), label='true')

    if model_from == 'keras':

        ax1.bar(classes, model.predict_proba(X[sample_idx, np.newaxis]).squeeze(), 

            label='prediction', color="red")

        prediction = model.predict_classes(X[sample_idx, np.newaxis])[0]

    else:

        ax1.bar(classes, model.forward(X[sample_idx]).squeeze(), 

            label='prediction', color="red")

        prediction = model.predict(X[sample_idx])

    ax1.set_xticks(classes)

    ax1.set_title(f'Output probabilities (prediction: {prediction})')

    ax1.set_xlabel('Digit class')

    ax1.legend()

    

def plot_history(history):

    """Plot the history of the training of a neural network."""

    fig = plt.figure(figsize=(10, 5))

    

    ax1 = fig.add_subplot(121)

    ax1.set(title='Model loss', xlabel='Epochs', ylabel='Loss')

    

    ax2 = fig.add_subplot(122)

    ax2.set(title='Model accuracy', xlabel='Epochs', ylabel='Accuracy')

    

    if len(history) == 2:

        ax1.plot(history['loss'])

        ax2.plot(history['acc'])

    else:

        for lr in history:

            ax1.plot(history[lr]['loss'], label=lr)

            ax2.plot(history[lr]['acc'], label=lr)

        ax1.legend(title='Learning rate')

        ax2.legend(title='Learning rate')

        

    return fig
# Load the data

digits_train = pd.read_csv('../input/train.csv')

digits_test = pd.read_csv('../input/test.csv')
# Define some global parameters

HEIGHT = 28 # Height of an image

WIDTH = 28 # Width of an image

PIXEL_NUMBER = 784 # Number of pixel in an image

PIXEL_VALUE = 255 # Maximum pixel value in an image
# Print an image

sample_index = 42



plot_mnist(digits_train, sample_index)

plt.show()
# Extract and convert the pixel as numpy array with dtype='float32'

train = np.asarray(digits_train.iloc[:, 1:], dtype='float32')

test = np.asarray(digits_test, dtype='float32')



train_target = np.asarray(digits_train.loc[:, 'label'], dtype='int32')
# Scale the data

scaler = preprocessing.StandardScaler()

train_scale = scaler.fit_transform(train)

test_scale = scaler.transform(test)
def one_hot(n_classes, y):

    """Encode categorical integer as a one-hot numerical array.

    

    Parameters

    ----------

    n_classes: integer

        The number of class considered

    y: integer array

        The integers that represent a category

    

    Return

    ------

    An array containing the one-hot encoding of the categorical

    array.

    

    Examples

    --------

    >>>one_hot(n_classes=10, y=3)

    array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])

    

    >>>one_hot(n_classes=10, y=[3, 0, 6])

    array([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],

           [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],

           [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]])

    """

    return np.eye(n_classes)[y]
def softmax(X):

    """ Compute the softmax function of a vector.

    

    Parameters

    ----------

    X: array

        Input values

    

    Return

    ------

    An array containing the softmax values. For each row, the

    sum of the column should be one.

    

    Examples

    --------

    >>>softmax([10, 2, -3])

    array([9.99662391e-01, 3.35349373e-04, 2.25956630e-06])

    >>>softmax([[10, 2, -3],

                [-1, 5, -20]])

    array([[9.99662391e-01, 3.35349373e-04, 2.25956630e-06],

       [2.47262316e-03, 9.97527377e-01, 1.38536042e-11]])

    """

    X = np.atleast_2d(X)

    

    exp = np.exp(X - np.max(X, axis=1, keepdims=True))

    return exp / np.sum(exp, axis=-1, keepdims=True)
# Test of the softmax function

print(f"Softmax of a single vector: {softmax([10, 2, -3])}.")
# Check that the probabilities sum to one

print(f"Sum of a softmax vector: {np.sum(softmax([10, 2, -3]))}")
# Test of the softmax function of two vectors

print(f"Softmax of two vectors: {softmax([[10, 2, -3], [-1, 5, -20]])}.")
# Check that the probabilities sum to one

print(f"Sum of a softmax vector: {np.sum(softmax([[10, 2, -3], [-1, 5, -20]]), axis=1)}")
def neg_log_likelihood(Y_true, Y_pred):

    """Compute the negative log-likelihood of a sample.

    

    Parameters

    ----------

    Y_true: array

        One-hot encoded class

    Y_pred: array

        Predicted probabilities

    

    Return

    ------

    The average negative log-likelihood of the sample. The closest we are

    to 0, the better are the predictions.

    

    Examples

    --------

    >>>neg_log_likelihood([1, 0, 0], [0.99, 0, 0.01])

    array([0.01005034])

    >>>neg_log_likelihood([1, 0, 0], [0.01, 0.98, 0.01])

    array([4.60517019])

    >>>Y_true = np.array([[0, 1, 0],

                          [1, 0, 0],

                          [0, 0, 1]])

    >>>Y_pred = np.array([[0,   1,    0],

                          [.99, 0.01, 0],

                          [0,   0,    1]])

    >>>neg_log_likelihood(Y_true, Y_pred)

    0.00335011195116715

    """

    Y_true = np.atleast_2d(Y_true)

    Y_pred = np.atleast_2d(Y_pred)

    

    loglike = np.sum(np.log(1e-8 + Y_pred) * Y_true, axis=1)

    return -np.mean(loglike)
# Make sure that it works for a simple sample at a time

print(f"Negative log-likelihood : {neg_log_likelihood([1, 0, 0], [.99, 0.01, 0])}.")
# Check that the negative log-likelihood of a very confident yet bad

# prediction is a much higher positive number

print(f"Negative log-likelihood : {neg_log_likelihood([1, 0, 0], [0.01, 0.01, .98])}.")
# Check that the average negative log-likelihood of the following three

# almost perfect predictions is close to 0

Y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

Y_pred = np.array([[0, 1, 0], [.99, 0.01, 0], [0, 0, 1]])



print(f"Negative log-likelihood : {neg_log_likelihood(Y_true, Y_pred)}.")
class LogisticRegression():

    """Define a class for the Logistic Regression with SGD optimization

    

    Parameters

    ----------

    input_size: integer

        Input size of the model (number of features)

    output_size: integer

        Output size of the model (number of classes)

        

    Arguments

    ---------

    W: array

        Array of weights

    b: vector

        Vector of bias

    output_size: integer

        Output size of the model (number of classes)

    """

    def __init__(self, input_size, output_size):

        self.W = np.random.uniform(size=(input_size, output_size),

                                   low=-0.1, high=0.1)

        self.b = np.random.uniform(size=output_size,

                                   low=-0.1, high=0.1)

        self.output_size = output_size

        

    def forward(self, X):

        """Compute the posterior probabilities

        

        Parameter

        ---------

        X: array, shape=(n_obs, n_features)

            Input array

            

        Return

        ------

        An array of shape `(n_obs, n_class)` of probabilities of each class.

        """

        Z = np.dot(X, self.W) + self.b

        return softmax(Z)

    

    def predict(self, X):

        """Give the most probable class of the observations

        

        Parameter

        ---------

        X: array, shape=(n_obs, n_features)

            Input array

        

        Return

        ------

        A vector of length `n_obs` that give the predicted class for

        each observation.

        """

        if len(X.shape) == 1:

            return np.argmax(self.forward(X))

        else:

            return np.argmax(self.forward(X), axis=1)

        

    def grad_loss(self, X, y_true):

        """Backpropagation of the gradients

        https://m2dsupsdlclass.github.io/lectures-labs/slides/02_backprop/index.html#40

        

        Parameters

        ----------

        X: array

            Observations

        y_true: integer

            True class of the observations

            

        Return

        ------

        Dictionnary containing the gradients for the weighs and the bias.

        """

        y_pred = self.forward(X)

        dnll_ouput = y_pred - one_hot(self.output_size, y_true)

        grad_W = np.outer(X, dnll_ouput)

        grad_b = dnll_ouput

        return {'W': grad_W, 'b': grad_b}

    

    def train(self, X, y, learning_rate=0.01):

        """Perform traditional SGD update without momentum.

        Update self.W and self.b

        

        Parameters

        ----------

        X: array

            Observations

        y: vector

            True classes

        learning_rate: double, default=0.01

            Learning rate

        """

        grads = self.grad_loss(X, y)

        self.W = self.W - learning_rate * grads['W']

        self.b = self.b - learning_rate * grads['b']

    

    def loss(self, X, y):

        """Compute the negative log-likelihood of the data.

        

        Parameters

        ----------

        X: array

            Observations

        y: vector

            True classes

            

        Return

        ------

        The average negative log-likelihood of the prediction of the observations.

        """

        return neg_log_likelihood(one_hot(self.output_size, y), self.forward(X))

    

    def accuracy(self, X, y):

        """Compute the prediction accuracy of the data.

        

        Parameters

        ----------

        X: array

            Observations

        y: vector

            True classes

            

        Return

        ------

        The accuracy of the predictions

        """

        y_preds = np.argmax(self.forward(X), axis=1)

        return np.mean(y_preds == y)
# Split the train set into train and validation set.

X_train, X_val, y_train, y_val = train_test_split(

    train_scale, train_target, test_size=0.15, random_state=42)
# Build a logistic model and test its forward inference

n_features = X_train.shape[1]

n_classes = len(np.unique(y_train))

lr = LogisticRegression(n_features, n_classes)



print(f"""Evaluation of the untrained model (recall that the weight and the bias are initialized as random):

    * train loss: {lr.loss(X_train, y_train)}

    * train accuracy: {lr.accuracy(X_train, y_train)}

    * test accuracy: {lr.accuracy(X_val, y_val)}

    """)
plot_prediction(X_val, y_val, lr, sample_idx=42)
learning_rate = 0.01

for i, (x, y) in enumerate(zip(X_train, y_train)):

    lr.train(x, y, learning_rate)

    if i % 10000 == 0:

        print(f"Update #{i}: train loss: {lr.loss(X_train, y_train)}, train accuracy: {lr.accuracy(X_train, y_train)}, test accuracy: {lr.accuracy(X_val, y_val)}")
plot_prediction(X_val, y_val, lr, sample_idx=42)
# Definition of the sigmoid function.

def sigmoid(X):

    """Compute the sigmoid function.

    

    Parameter

    ---------

    X: array-like

        Input array

        

    Return

    ------

    Sigmoid function on the input array.

    

    Examples

    --------

    >>>sigmoid(np.array([5, 2, 0]))

    array([0.99330715, 0.88079708, 0.5])

    >>>sigmoid(np.array([[5, 2, 0], [-2, 3, 0]]))

    array([[0.99330715, 0.88079708, 0.5],

           [0.11920292, 0.95257413, 0.5]])

           

    Notes

    -----

    X < -709 is to prevent exp overflow.

    """

    X[X < -709] = -709

    return 1 / (1 + np.exp(-X))



def dsigmoid(X):

    """Compute the element-wide derivative of the sigmoid function.

    

    Parameter

    ---------

    X: array-like

        Input array

    

    Return

    ------

    Derivative of the sigmoid function on the input array.

    

    Examples

    --------

    >>>dsigmoid(np.array([5, 2, 0]))

    array([0.00664806, 0.10499359, 0.25])

    >>>dsigmoid(np.array([[5, 2, 0], [-2, 3, 0]]))

    array([[0.00664806, 0.10499359, 0.25],

           [0.10499359, 0.04517666, 0.25]])

    """

    S = sigmoid(X)

    return S * (1 - S)
# Plot of the functions

X = np.linspace(-5, 5, 100)

plt.plot(X, sigmoid(X), label='Sigmoid')

plt.plot(X, dsigmoid(X), label='Derivative')

plt.legend(loc='best')

plt.show()
# Definition of the tanh function.

def tanh(X):

    """Compute the tanh function.

    

    Parameter

    ---------

    X: array-like

        Input array

        

    Return

    ------

    Tanh function on the input array.

    

    Examples

    --------

    >>>tanh(np.array([5, 2, 0]))

    array([0.9999092 , 0.96402758, 0.])

    >>>tanh(np.array([[5, 2, 0], [-2, 3, 0]]))

    array([[ 0.9999092 ,  0.96402758,  0.],

           [-0.96402758,  0.99505475,  0.]])

    

    Notes

    -----

    X > 354 is to prevent exp overflow.

    """

    X[X > 354] = 354

    exp = np.exp(2*X)

    return (exp - 1) / (exp + 1)



def dtanh(X):

    """Compute the element-wide derivative of the tanh function.

    

    Parameter

    ---------

    X: array-like

        Input array

    

    Return

    ------

    Derivative of the tanh function on the input array.

    

    Examples

    --------

    >>>dtanh(np.array([5, 2, 0]))

    array([1.81583231e-04, 7.06508249e-02, 1.00000000e+00])

    >>>dtanh(np.array([[5, 2, 0], [-2, 3, 0]]))

    array([[1.81583231e-04, 7.06508249e-02, 1.00000000e+00],

           [7.06508249e-02, 9.86603717e-03, 1.00000000e+00]])

    """

    S = tanh(X)

    return 1 - np.power(S, 2)
# Plot of the functions

X = np.linspace(-5, 5, 100)

plt.plot(X, tanh(X), label='Tanh')

plt.plot(X, dtanh(X), label='Derivative')

plt.legend(loc='best')

plt.show()
# Definition of the relu function.

def relu(X):

    """Compute the relu function.

    

    Parameter

    ---------

    X: array-like

        Input array

        

    Return

    ------

    Relu function on the input array.

    

    Examples

    --------

    >>>relu(np.array([5, -2, 0]))

    array([5, 0, 0])

    >>>relu(np.array([[5, 2, 0], [-2, 3, 0]]))

    array([[5, 2, 0],

           [0, 3, 0]])

    """

    return np.maximum(0, X).astype(np.float64)



def drelu(X):

    """Compute the element-wide derivative of the relu function.

    

    Parameter

    ---------

    X: array-like

        Input array

    

    Return

    ------

    Derivative of the relu function on the input array.

    

    Examples

    --------

    >>>drelu(np.array([5, -2, 0]))

    array([1., 0., 0.])

    >>>drelu(np.array([[5, 2, 0], [-2, 3, 0]]))

    array([[1., 1., 0.],

           [0., 1., 0.]])

    """

    return np.array(X > 0).astype(np.float64)
# Plot of the functions

X = np.linspace(-5, 5, 100)

plt.plot(X, relu(X), label='Relu')

plt.plot(X, drelu(X), label='Derivative')

plt.legend(loc='best')

plt.show()
EPSILON = 1e-8



class NeuralNet():

    """Define a class for the MultiLayer Perceptron with 

    one hidden layer with a sigmoid activation function.

    

    Parameters

    ----------

    input_size: integer

        Input size of the model (number of features)

    hidden_size: integer

        Size of the hidden layer (hyperparameter)

    output_size: integer

        Output size of the model (number of classes)

    activation_function: 'sigmoid', 'tanh' or 'relu', default='sigmoid'

        Activation function for the hidden layer.

    Arguments

    ---------

    W_h: array

        Weight array of the hidden layer

    W_o: array

        Weight array of the output layer

    b_h: vector

        Bias vecor of the hidden layer

    b_o: vector

        Bias vector of the output layer

    output_size: integer

        Output size of the model (number of classes)

    """

    def __init__(self, input_size, hidden_size, output_size, activation_function='sigmoid'):

        self.W_h = np.random.uniform(size=(input_size, hidden_size),

                                     low=-0.1, high=0.1)

        self.b_h = np.zeros(hidden_size)

        self.W_o = np.random.uniform(size=(hidden_size, output_size),

                                     low=-0.1, high=0.1)

        self.b_o = np.zeros(output_size)

        self.output_size = output_size

        self.activation_function = activation_function

        

    def forward(self, X):

        """Compute the posterior probabilities.

        

        Parameter

        ---------

        X: array, shape=(n_obs, n_features)

            Input array

        

        Return

        ------

        An array of shape `(n_obs, n_class)` of probabilities of each class.

        """

        if self.activation_function == 'tanh':

            h = tanh(np.dot(X, self.W_h) + self.b_h)

        elif self.activation_function == 'relu':

            h = relu(np.dot(X, self.W_h) + self.b_h)

        else:

            h = sigmoid(np.dot(X, self.W_h) + self.b_h)

        y = softmax(np.dot(h, self.W_o) + self.b_o)

        return y

    

    def forward_keep_activations(self, X):

        """Compute the posterior probabilities.

        

        Parameter

        ---------

        X: array, shape=(n_obs, n_features)

            Input array

            

        Return

        ------

        An array of shape `(n_obs, n_class)` of probabilities of each class,

        and the hidden activations and pre-activitations arrays.

        """

        z_h = np.dot(X, self.W_h) + self.b_h

        if self.activation_function == 'tanh':

            h = tanh(z_h)

        elif self.activation_function == 'relu':

            h = relu(z_h)

        else:

            h = sigmoid(z_h)

        y = softmax(np.dot(h, self.W_o) + self.b_o)

        return y, h, z_h

        

    def grad_loss(self, X, y_true):

        """Backpropagation of the gradients

        https://m2dsupsdlclass.github.io/lectures-labs/slides/02_backprop/index.html#40

        

        Parameters

        ----------

        X: array

            Observations

        y_true: integer

            True class of the observations

            

        Return

        ------

        Dictionary containing the gradients for the weights and the bias.

        """

        y_pred, h, z_h = self.forward_keep_activations(X)

        grad_z_o = y_pred - one_hot(self.output_size, y_true)

        

        grad_W_o = np.outer(h, grad_z_o)

        grad_b_o = grad_z_o

        

        grad_h = np.dot(grad_z_o, np.transpose(self.W_o))

        if self.activation_function == 'tanh':

            grad_z_h = grad_h * dtanh(z_h)

        elif self.activation_function == 'relu':

            grad_z_h = grad_h * drelu(z_h)

        else:

            grad_z_h = grad_h * dsigmoid(z_h)

        

        grad_W_h = np.outer(X, grad_z_h)

        grad_b_h = grad_z_h

        

        return {"W_h": grad_W_h, "b_h": grad_b_h, "W_o": grad_W_o, "b_o": grad_b_o}

    

    def train(self, X, y, learning_rate):

        """Perform traditional SGD update without momentum.

        Update self.W_h, self.W_o, self.b_h and self.b_o.

        

        Parameters

        ----------

        X: array

            Observations

        y: vector

            True classes

        learning_rate: double, default=0.01

            Learning rate

        """

        grads = self.grad_loss(X, y)

        self.W_h = self.W_h - learning_rate * grads['W_h']

        self.b_h = self.b_h - learning_rate * grads['b_h']

        

        self.W_o = self.W_o - learning_rate * grads['W_o']

        self.b_o = self.b_o - learning_rate * grads['b_o']

    

    def predict(self, X):

        """Give the most probable class of the observations.

        

        Parameter

        ---------

        X: array, shape=(n_obs, n_features)

            Input array

        

        Return

        ------

        A vector of length `n_obs` that give the predicted class for each observation.

        """

        if len(X.shape) == 1:

            return np.argmax(self.forward(X))

        else:

            return np.argmax(self.forward(X), axis=1)



    def loss(self, X, y):

        """Compute the negative log-likelihood of the data.

        

        Parameters

        ----------

        X: array

            Observations

        y: vector

            True classes

            

        Return

        ------

        The average negative log-likelihood of the prediction of the observations.

        """

        return neg_log_likelihood(one_hot(self.output_size, y), self.forward(X))

    

    def accuracy(self, X, y):

        """Compute the prediction accuracy of the data.

        

        Parameters

        ----------

        X: array

            Observations

        y: vector

            True classes

        

        Return

        ------

        The accuracy of the predictions

        """

        y_preds = np.argmax(self.forward(X), axis=1)

        return np.mean(y_preds == y)
# Build a NeuralNet model and test its forward inference

n_features = X_train.shape[1]

n_classes = len(np.unique(y_train))

n_hidden = 10

model = NeuralNet(n_features, n_hidden, n_classes)



print(f"""Evaluation of the untrained model (recall that the weights are initialized as random):

    * train loss: {model.loss(X_train, y_train)}

    * train accuracy: {model.accuracy(X_train, y_train)}

    * test accuracy: {model.accuracy(X_val, y_val)}

    """)
plot_prediction(X_val, y_val, model, sample_idx=42)
# Train the model for some epochs

EPOCHS = 15

learning_rate = 0.1



losses = [model.loss(X_train, y_train)]

accuracies = [model.accuracy(X_train, y_train)]

accuracies_val = [model.accuracy(X_val, y_val)]



for epoch in range(EPOCHS):

    for i, (x, y) in enumerate(zip(X_train, y_train)):

        model.train(x, y, learning_rate)

    losses.append(model.loss(X_train, y_train))

    accuracies.append(model.accuracy(X_train, y_train))

    accuracies_val.append(model.accuracy(X_val, y_val))

    print(f"Epoch #{epoch+1}: train loss: {losses[-1]}, train accuracy: {accuracies[-1]}, test accuracy: {accuracies_val[-1]}")
# Plot losses and accuracies.

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))



ax0.plot(losses, label='Loss')

ax0.set_title('Model Loss')

ax0.set_xlabel('Epochs')

ax0.legend()



ax1.plot(accuracies, label='Train accuracy')

ax1.plot(accuracies_val, label='Test accuracy')

ax1.set_title('Model accuracy')

ax1.set_ylim(0, 1.1)

ax1.set_xlabel('Epochs')

ax1.legend(loc='best')



plt.show()
plot_prediction(X_val, y_val, model, sample_idx=42)
# The loss is the negative log-likelihood for each picture.

val_losses = -np.sum(np.log(1e-8 + model.forward(X_val)) * one_hot(10, y_val), axis=1)



# We, then, sort them by ascending loss.

rank_loss = val_losses.argsort()



# Plot the top 5 worst predictions

for idx in rank_loss[-5:]:

    plot_prediction(X_val, y_val, model, sample_idx=idx)
n_features = X_train.shape[1]

n_classes = len(np.unique(y_train))

n_hidden = 10



EPOCHS = 15

learning_rates = np.logspace(-5, 1, num=7)



losses, accuracies, accuracies_val = {}, {}, {}

for lr in learning_rates:

    model = NeuralNet(n_features, n_hidden, n_classes)



    losses[lr] = [model.loss(X_train, y_train)]

    accuracies[lr] = [model.accuracy(X_train, y_train)]

    accuracies_val[lr] = [model.accuracy(X_val, y_val)]

    for epoch in range(EPOCHS):

        for i, (x, y) in enumerate(zip(X_train, y_train)):

            model.train(x, y, lr)

        losses[lr].append(model.loss(X_train, y_train))

        accuracies[lr].append(model.accuracy(X_train, y_train))

        accuracies_val[lr].append(model.accuracy(X_val, y_val))
# Plot losses and accuracies.

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(20,8))



colors = list('rgbcmyk')

for lr, loss in losses.items():

    ax0.plot(loss, label=f'{lr}', color=colors.pop())

ax0.set_title('Model Loss')

ax0.set_xlabel('Epochs')

ax0.legend(loc='best', title='Learning rate')



colors = list('rgbcmyk')

for lr, accuracy in accuracies.items():

    ax1.plot(accuracy, label=f'{lr}', color=colors.pop())

ax1.set_title('Model train accuracy')

ax1.set_ylim(0, 1.1)

ax1.set_xlabel('Epochs')

ax1.legend(loc='best', title='Learning rate')



colors = list('rgbcmyk')

for lr, accuracy in accuracies_val.items():

    ax2.plot(accuracy, label=f'{lr}', color=colors.pop())

ax2.set_title('Model test accuracy')

ax2.set_ylim(0, 1.1)

ax2.set_xlabel('Epochs')

ax2.legend(loc='best', title='Learning rate')



plt.show()
n_features = X_train.shape[1]

n_classes = len(np.unique(y_train))

n_hidden = np.array([5, 10, 25, 50, 100])



EPOCHS = 15

learning_rates = 0.1



losses, accuracies, accuracies_val = {}, {}, {}

for n in n_hidden:

    model = NeuralNet(n_features, n, n_classes)



    losses[n] = [model.loss(X_train, y_train)]

    accuracies[n] = [model.accuracy(X_train, y_train)]

    accuracies_val[n] = [model.accuracy(X_val, y_val)]

    for epoch in range(EPOCHS):

        for i, (x, y) in enumerate(zip(X_train, y_train)):

            model.train(x, y, learning_rate)

        losses[n].append(model.loss(X_train, y_train))

        accuracies[n].append(model.accuracy(X_train, y_train))

        accuracies_val[n].append(model.accuracy(X_val, y_val))
# Plot losses and accuracies.

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(20,8))



colors = list('rgbcmyk')

for n_hidden, loss in losses.items():

    ax0.plot(loss, label=f'{n_hidden}', color=colors.pop())

ax0.set_title('Model Loss')

ax0.set_xlabel('Epochs')

ax0.legend(loc='best', title='Hidden size')



colors = list('rgbcmyk')

for n_hidden, accuracy in accuracies.items():

    ax1.plot(accuracy, label=f'{n_hidden}', color=colors.pop())

ax1.set_title('Model train accuracy')

ax1.set_ylim(0, 1.1)

ax1.set_xlabel('Epochs')

ax1.legend(loc='best', title='Hidden size')



colors = list('rgbcmyk')

for n_hidden, accuracy in accuracies_val.items():

    ax2.plot(accuracy, label=f'{n_hidden}', color=colors.pop())

ax2.set_title('Model test accuracy')

ax2.set_ylim(0, 1.1)

ax2.set_xlabel('Epochs')

ax2.legend(loc='best', title='Hidden size')



plt.show()
# Build a NeuralNet model and test its forward inference with tanh activation function

n_features = X_train.shape[1]

n_classes = len(np.unique(y_train))

n_hidden = 10

model = NeuralNet(n_features, n_hidden, n_classes, activation_function='tanh')



print(f"""Evaluation of the untrained model (recall that the weights are initialized as random):

    * train loss: {model.loss(X_train, y_train)}

    * train accuracy: {model.accuracy(X_train, y_train)}

    * test accuracy: {model.accuracy(X_val, y_val)}

    """)
# Train the model for some epochs

EPOCHS = 15

learning_rate = 0.1



losses = [model.loss(X_train, y_train)]

accuracies = [model.accuracy(X_train, y_train)]

accuracies_val = [model.accuracy(X_val, y_val)]



for epoch in range(EPOCHS):

    for i, (x, y) in enumerate(zip(X_train, y_train)):

        model.train(x, y, learning_rate)

    losses.append(model.loss(X_train, y_train))

    accuracies.append(model.accuracy(X_train, y_train))

    accuracies_val.append(model.accuracy(X_val, y_val))

    print(f"Epoch #{epoch+1}: train loss: {losses[-1]}, train accuracy: {accuracies[-1]}, test accuracy: {accuracies_val[-1]}")
# Plot losses and accuracies.

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))



ax0.plot(losses, label='Loss')

ax0.set_title('Model Loss')

ax0.set_xlabel('Epochs')

ax0.legend()



ax1.plot(accuracies, label='Train accuracy')

ax1.plot(accuracies_val, label='Test accuracy')

ax1.set_title('Model accuracy')

ax1.set_ylim(0, 1.1)

ax1.set_xlabel('Epochs')

ax1.legend(loc='best')



plt.show()
# Build a NeuralNet model and test its forward inference with relu activation function

n_features = X_train.shape[1]

n_classes = len(np.unique(y_train))

n_hidden = 10

model = NeuralNet(n_features, n_hidden, n_classes, activation_function='relu')



print(f"""Evaluation of the untrained model (recall that the weights are initialized as random):

    * train loss: {model.loss(X_train, y_train)}

    * train accuracy: {model.accuracy(X_train, y_train)}

    * test accuracy: {model.accuracy(X_val, y_val)}

    """)
# Train the model for some epochs

EPOCHS = 15

learning_rate = 0.01



losses = [model.loss(X_train, y_train)]

accuracies = [model.accuracy(X_train, y_train)]

accuracies_val = [model.accuracy(X_val, y_val)]



for epoch in range(EPOCHS):

    for i, (x, y) in enumerate(zip(X_train, y_train)):

        model.train(x, y, learning_rate)

    losses.append(model.loss(X_train, y_train))

    accuracies.append(model.accuracy(X_train, y_train))

    accuracies_val.append(model.accuracy(X_val, y_val))

    print(f"Epoch #{epoch+1}: train loss: {losses[-1]}, train accuracy: {accuracies[-1]}, test accuracy: {accuracies_val[-1]}")
# Plot losses and accuracies.

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))



ax0.plot(losses, label='Loss')

ax0.set_title('Model Loss')

ax0.set_xlabel('Epochs')

ax0.legend()



ax1.plot(accuracies, label='Train accuracy')

ax1.plot(accuracies_val, label='Test accuracy')

ax1.set_title('Model accuracy')

ax1.set_ylim(0, 1.1)

ax1.set_xlabel('Epochs')

ax1.legend(loc='best')



plt.show()
EPSILON = 1e-8



FUNCTION = {'sigmoid': sigmoid,

            'tanh': tanh,

            'relu': relu}

DERIVATIVE = {'sigmoid': dsigmoid,

              'tanh': dtanh,

              'relu': drelu}



class NeuralNet():

    """Define a class for the MultiLayer Perceptron with 

    multiple hidden layer.

    

    Parameters

    ----------

    sizes: list of integer

        Contains the number of neurons in the respective layers of the network.

    activation_function: str, 'sigmoid', 'tanh' or 'relu', default='sigmoid'

        Activation function for the hidden layer.

    Arguments

    ---------

    num_layers: integer

        Number of layers in the network

    sizes: list of integer

        Contains the number of neurons in the respective layers of the network.

    weights: list of arrays

        Weight arrays

    biases: vector

        Bias vecor of the hidden layer

    activation_function: 'sigmoid', 'tanh' or 'relu'

        Activation function for the hidden layers

    deriv_activation: 'dsigmoid', 'dtanh' or 'drelu'

    """

    def __init__(self, sizes, activation_function='sigmoid'):

        self.num_layers = len(sizes)

        self.sizes = sizes

        self.biases = [np.zeros(shape=(output_size, 1)) 

                       for output_size in sizes[1:]]

        self.weights = [np.random.uniform(size=(output_size, input_size), 

                                          low=-0.1, high=0.1) 

                        for input_size, output_size in zip(sizes[:-1], sizes[1:])]

        self.activation_function = FUNCTION.get(activation_function, sigmoid)

        self.deriv_activation = DERIVATIVE.get(activation_function, dsigmoid)

        

    def forward(self, X):

        """Compute the posterior probabilities.

        

        Parameter

        ---------

        X: array, shape=(n_obs, n_features)

            Input array

        

        Return

        ------

        An array of shape `(n_obs, n_classes)` of probabilities of each class.

        """

        A = np.transpose(X)

        for bias, weight in zip(self.biases[:-1], self.weights[:-1]):

            A = self.activation_function(np.dot(weight, A) + bias)

        

        A = np.dot(self.weights[-1], A) + self.biases[-1]

        y = softmax(np.transpose(A))

        return y

    

    def forward_keep_activations(self, X):

        """Compute the posterior probabilities.

        

        Parameter

        ---------

        X: array, shape=(n_obs, n_features)

            Input array

            

        Return

        ------

        A list containing :

            - an array of shape `(n_obs, n_class)` of probabilities of each class

            - a list of array of the hidden activations 

            - a list of array of the pre-activitations

        """

        A = np.transpose(X)

        z_h = []

        h = [A]

        

        for bias, weight in zip(self.biases[:-1], self.weights[:-1]):

            z_h_ = np.dot(weight, A) + bias

            A = self.activation_function(z_h_)

            z_h.append(z_h_)

            h.append(A)

            

        z_h_ = np.dot(self.weights[-1], A) + self.biases[-1]

        A = softmax(np.transpose(z_h_))

        z_h.append(z_h_)

        h.append(np.transpose(A))

        y = np.transpose(h[-1])

        return y, h, z_h

        

    def grad_loss(self, X, y_true):

        """Backpropagation of the gradients

        https://m2dsupsdlclass.github.io/lectures-labs/slides/02_backprop/index.html#41

        

        nabla_b and nabla_w are layer-by-layer lists of numpy arrays, similar to

        self.biaises and self.weight.

        

        Parameters

        ----------

        X: array

            Observations

        y_true: integer

            True class of the observations

            

        Return

        ------

        Tuple containing the gradients for the weights and the bias.

        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]

        nabla_w = [np.zeros(w.shape) for w in self.weights]

        

        # Feed-forward

        y_pred, h, z_h = self.forward_keep_activations(X)

        

        # Backward

        delta = np.transpose(y_pred - one_hot(self.sizes[-1], y_true))

        nabla_b[-1] = delta

        nabla_w[-1] = np.dot(delta, np.transpose(h[-2]))

        

        for l in range(2, self.num_layers):

            grad_h = np.dot(np.transpose(self.weights[-l + 1]), delta)

            delta = grad_h * self.deriv_activation(z_h[-l])

            nabla_b[-l] = delta

            nabla_w[-l] = np.dot(delta, np.transpose(h[-l-1]))



        return nabla_b, nabla_w

    

    def train(self, X, y, learning_rate):

        """Perform traditional SGD update without momentum.

        Update self.weights and self.biases

        

        Parameters

        ----------

        X: array

            Observations

        y: vector

            True classes

        learning_rate: double, default=0.01

            Learning rate

        """

        nabla_b, nabla_w = self.grad_loss(X, y)

        

        self.biases = [b - learning_rate * nb for b, nb in zip(self.biases, nabla_b)]

        self.weights = [w - learning_rate * nw for w, nw in zip(self.weights, nabla_w)]

    

    def predict(self, X):

        """Give the most probable class of the observations.

        

        Parameter

        ---------

        X: array, shape=(n_obs, n_features)

            Input array

        

        Return

        ------

        A vector of length `n_obs` that give the predicted class for each observation.

        """

        if len(X.shape) == 1:

            return np.argmax(self.forward(X))

        else:

            return np.argmax(self.forward(X), axis=1)



    def loss(self, X, y):

        """Compute the negative log-likelihood of the data.

        

        Parameters

        ----------

        X: array

            Observations

        y: vector

            True classes

            

        Return

        ------

        The average negative log-likelihood of the prediction of the observations.

        """

        return neg_log_likelihood(one_hot(self.sizes[-1], y), self.forward(X))



    def accuracy(self, X, y):

        """Compute the prediction accuracy of the data.

        

        Parameters

        ----------

        X: array

            Observations

        y: vector

            True classes

        

        Return

        ------

        The accuracy of the predictions

        """

        y_preds = np.argmax(self.forward(X), axis=1)

        return np.mean(y_preds == y)

# Build a NeuralNet model and test its forward inference with 2 hidden layers

n_features = X_train.shape[1]

n_classes = len(np.unique(y_train))

n_hidden = 20

n_hidden2 = 10

sizes = [n_features, n_hidden, n_hidden2, n_classes]

model = NeuralNet(sizes, activation_function='sigmoid')



print(f"""Evaluation of the untrained model (recall that the weights are initialized as random):

    * train loss: {model.loss(X_train, y_train)}

    * train accuracy: {model.accuracy(X_train, y_train)}

    * test accuracy: {model.accuracy(X_val, y_val)}

    """)
# Train the model for some epochs

EPOCHS = 15

learning_rate = 0.1



losses = [model.loss(X_train, y_train)]

accuracies = [model.accuracy(X_train, y_train)]

accuracies_val = [model.accuracy(X_val, y_val)]



for epoch in range(EPOCHS):

    for i, (x, y) in enumerate(zip(X_train, y_train)):

        model.train(x[np.newaxis], y, learning_rate)

    losses.append(model.loss(X_train, y_train))

    accuracies.append(model.accuracy(X_train, y_train))

    accuracies_val.append(model.accuracy(X_val, y_val))

    print(f"Epoch #{epoch+1}: train loss: {losses[-1]}, train accuracy: {accuracies[-1]}, test accuracy: {accuracies_val[-1]}")
# Plot losses and accuracies.

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))



ax0.plot(losses, label='Loss')

ax0.set_title('Model Loss')

ax0.set_xlabel('Epochs')

ax0.legend()



ax1.plot(accuracies, label='Train accuracy')

ax1.plot(accuracies_val, label='Test accuracy')

ax1.set_title('Model accuracy')

ax1.set_ylim(0, 1.1)

ax1.set_xlabel('Epochs')

ax1.legend(loc='best')



plt.show()
EPSILON = 1e-8



FUNCTION = {'sigmoid': sigmoid,

            'tanh': tanh,

            'relu': relu}

DERIVATIVE = {'sigmoid': dsigmoid,

              'tanh': dtanh,

              'relu': drelu}



class NeuralNet():

    """Define a class for the MultiLayer Perceptron with 

    multiple hidden layer.

    

    Parameters

    ----------

    sizes: list of integer

        Contains the number of neurons in the respective layers of the network.

    activation_function: str, 'sigmoid', 'tanh' or 'relu', default='sigmoid'

        Activation function for the hidden layer.

    Arguments

    ---------

    num_layers: integer

        Number of layers in the network

    sizes: list of integer

        Contains the number of neurons in the respective layers of the network.

    weights: list of arrays

        Weight arrays

    biases: vector

        Bias vecor of the hidden layer

    activation_function: 'sigmoid', 'tanh' or 'relu'

        Activation function for the hidden layers

    deriv_activation: 'dsigmoid', 'dtanh' or 'drelu'

    """

    def __init__(self, sizes, biases=None, weights=None, activation_function='sigmoid'):

        self.num_layers = len(sizes)

        self.sizes = sizes

        if biases is None:

            self.biases = [np.zeros(shape=(output_size, 1)) for output_size in sizes[1:]]

        else:

            self.biases = biases

        if weights is None:

            self.weights = [np.random.uniform(size=(output_size, input_size), low=-0.1, high=0.1) 

                        for input_size, output_size in zip(sizes[:-1], sizes[1:])]

        else:

            self.weights = weights

        self.activation_function = FUNCTION.get(activation_function, sigmoid)

        self.deriv_activation = DERIVATIVE.get(activation_function, dsigmoid)

        

    def forward(self, X):

        """Compute the posterior probabilities.

        

        Parameter

        ---------

        X: array, shape=(n_obs, n_features)

            Input array

        

        Return

        ------

        An array of shape `(n_obs, n_classes)` of probabilities of each class.

        """

        A = np.transpose(X)

        for bias, weight in zip(self.biases[:-1], self.weights[:-1]):

            A = self.activation_function(np.dot(weight, A) + bias)

        

        A = np.dot(self.weights[-1], A) + self.biases[-1]

        y = softmax(np.transpose(A))

        return y

    

    def forward_keep_activations(self, X):

        """Compute the posterior probabilities.

        

        Parameter

        ---------

        X: array, shape=(n_obs, n_features)

            Input array

            

        Return

        ------

        A list containing :

            - an array of shape `(n_obs, n_class)` of probabilities of each class

            - a list of array of the hidden activations 

            - a list of array of the pre-activitations

        """

        A = np.transpose(X)

        z_h = []

        h = [A]

        

        for bias, weight in zip(self.biases[:-1], self.weights[:-1]):

            z_h_ = np.dot(weight, A) + bias

            A = self.activation_function(z_h_)

            z_h.append(z_h_)

            h.append(A)

            

        z_h_ = np.dot(self.weights[-1], A) + self.biases[-1]

        A = softmax(np.transpose(z_h_))

        z_h.append(z_h_)

        h.append(np.transpose(A))

        y = np.transpose(h[-1])

        return y, h, z_h

        

    def grad_loss(self, X, y_true):

        """Backpropagation of the gradients

        https://m2dsupsdlclass.github.io/lectures-labs/slides/02_backprop/index.html#41

        

        nabla_b and nabla_w are layer-by-layer lists of numpy arrays, similar to

        self.biaises and self.weight.

        

        Parameters

        ----------

        X: array

            Observations

        y_true: integer

            True class of the observations

            

        Return

        ------

        Tuple containing the gradients for the weights and the bias.

        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]

        nabla_w = [np.zeros(w.shape) for w in self.weights]

        

        # Feed-forward

        y_pred, h, z_h = self.forward_keep_activations(X)

        

        # Backward

        delta = np.transpose(y_pred - one_hot(self.sizes[-1], y_true))

        nabla_b[-1] = delta

        nabla_w[-1] = np.dot(delta, np.transpose(h[-2]))

        

        for l in range(2, self.num_layers):

            grad_h = np.dot(np.transpose(self.weights[-l + 1]), delta)

            delta = grad_h * self.deriv_activation(z_h[-l])

            nabla_b[-l] = delta

            nabla_w[-l] = np.dot(delta, np.transpose(h[-l-1]))



        return nabla_b, nabla_w

    

    def train(self, X, Y, learning_rate):

        """Perform traditional SGD update without momentum.

        Update self.weights and self.biases

        

        Parameters

        ----------

        X: array

            Observations

        y: vector

            True classes

        learning_rate: double, default=0.01

            Learning rate

        """        

        nabla_b = [np.zeros(b.shape) for b in self.biases]

        nabla_w = [np.zeros(w.shape) for w in self.weights]

        

        for x, y in zip(X, Y):

            delta_nabla_b, delta_nabla_w = self.grad_loss(x[np.newaxis], y)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        

        self.weights = [w - (learning_rate / len(Y)) * nw 

                        for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (learning_rate / len(Y)) * nb 

                       for b, nb in zip(self.biases, nabla_b)]



    def predict(self, X):

        """Give the most probable class of the observations.

        

        Parameter

        ---------

        X: array, shape=(n_obs, n_features)

            Input array

        

        Return

        ------

        A vector of length `n_obs` that give the predicted class for each observation.

        """

        if len(X.shape) == 1:

            return np.argmax(self.forward(X))

        else:

            return np.argmax(self.forward(X), axis=1)



    def loss(self, X, y):

        """Compute the negative log-likelihood of the data.

        

        Parameters

        ----------

        X: array

            Observations

        y: vector

            True classes

            

        Return

        ------

        The average negative log-likelihood of the prediction of the observations.

        """

        return neg_log_likelihood(one_hot(self.sizes[-1], y), self.forward(X))



    def accuracy(self, X, y):

        """Compute the prediction accuracy of the data.

        

        Parameters

        ----------

        X: array

            Observations

        y: vector

            True classes

        

        Return

        ------

        The accuracy of the predictions

        """

        y_preds = np.argmax(self.forward(X), axis=1)

        return np.mean(y_preds == y)

# Build a NeuralNet model and test its forward inference with 2 hidden layers

n_features = X_train.shape[1]

n_classes = len(np.unique(y_train))

n_hidden = 100

sizes = [n_features, n_hidden, n_classes]

model = NeuralNet(sizes, activation_function='sigmoid')



print(f"""Evaluation of the untrained model (recall that the weights are initialized as random):

    * train loss: {model.loss(X_train, y_train)}

    * train accuracy: {model.accuracy(X_train, y_train)}

    * test accuracy: {model.accuracy(X_val, y_val)}

    """)
# Train the model for some epochs

EPOCHS = 15

learning_rate = 0.1

mini_batch_size = 32



losses = [model.loss(X_train, y_train)]

accuracies = [model.accuracy(X_train, y_train)]

accuracies_val = [model.accuracy(X_val, y_val)]



for epoch in range(EPOCHS):

    mini_batches = [(X_train[k:k + mini_batch_size], y_train[k:k + mini_batch_size]) 

                    for k in range(0, len(X_train), mini_batch_size)]

    for batch in mini_batches:

        model.train(batch[0], batch[1], learning_rate)

    losses.append(model.loss(X_train, y_train))

    accuracies.append(model.accuracy(X_train, y_train))

    accuracies_val.append(model.accuracy(X_val, y_val))

    print(f"Epoch #{epoch+1}: train loss: {losses[-1]}, train accuracy: {accuracies[-1]}, test accuracy: {accuracies_val[-1]}")
# Plot losses and accuracies.

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))



ax0.plot(losses, label='Loss')

ax0.set_title('Model Loss')

ax0.set_xlabel('Epochs')

ax0.legend()



ax1.plot(accuracies, label='Train accuracy')

ax1.plot(accuracies_val, label='Test accuracy')

ax1.set_title('Model accuracy')

ax1.set_ylim(0, 1.1)

ax1.set_xlabel('Epochs')

ax1.legend(loc='best')



plt.show()
# Encoded the target vector as one-hot-encoding vector.

target = to_categorical(y_train)

target_val = to_categorical(y_val)
# Define some parameters

N = X_train.shape[1]

H1 = 100 # Dimension of the first hidden layer

H2 = 20 # Dimension of the second hidden layer

K = 10 # Dimension of output layer (number of classes to predict)

lr = 0.1 # Learning rate for the loss function

epochs = 15 # Number of epochs for the NN

batch_size = 32 # Size of the batch



# Define the model

model = Sequential()

model.add(Dense(H1, input_dim=N, activation='sigmoid'))

model.add(Dense(H2, input_dim=H1, activation='sigmoid'))

model.add(Dense(K, input_dim=H2, activation='softmax'))



# Print the model

model.summary()
# Define the loss function with the optimizer

model.compile(optimizer=optimizers.SGD(lr=lr),

             loss='categorical_crossentropy',

             metrics=['accuracy'])



# Fit the model

history = model.fit(X_train, target, epochs=epochs, batch_size=batch_size, verbose=0)
plot_history(history.history)

plt.show()
print(f"""The negative log-likelihood of a sample 42 in the test set is \

{neg_log_likelihood(target_val[42], model.predict_proba(X_val[42, np.newaxis]))}.""")
plot_prediction(X_val, y_val, model, sample_idx=42, model_from='keras')
print(f"""The negative log-likelihood on the full test set is \

{neg_log_likelihood(target_val, model.predict_proba(X_val))}.""")
print(f"""The negative log-likelihood on the full train set is \

{neg_log_likelihood(target, model.predict_proba(X_train))}.""")