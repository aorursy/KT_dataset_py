import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# activation functions and derivatives
def ReLU(x, deriv=False):
    if deriv:
        return 1. * (x > 0)
    else:
        return x * (x > 0)

def LReLU(x, deriv=False, alpha=0.001):
    if deriv:
        return 1. * (x >= 0) + alpha * (x < 0)
    else:
        return x * (x >= 0) + (alpha * x) * (x < 0) 
    
def LReLU6(x, deriv=False, alpha=0.1):
    x = np.clip(x, -6, 6)
    if deriv:
        return 1. * (x >= 0) + alpha * (x < 0)
    else:
        return x * (x >= 0) + (alpha * x) * (x < 0) 

def sigmoid(x, deriv=False):
    if deriv:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        x = np.clip(x, -700, 700)
        return 1 / (1 + np.exp(-x))

def tanh(x, deriv=False):
    if deriv:
        return 1.0 - (np.tanh(x) ** 2)
    else:
        return np.tanh(x)

def linear(x, deriv=False, alpha=1.0):
    if deriv:
        return alpha
    else:
        return alpha * x
def split_to_minibatch(z, batch_size=32):
    z_length = len(z)
    current_batch_size = batch_size
    batch_split_points = []
    while current_batch_size < z_length:
        batch_split_points.append(current_batch_size)
        current_batch_size += batch_size
    return np.split(z, batch_split_points)

def shuffle_split(X, y, test_split=0.2):
    X, y = shuffle(X, y)
    return train_test_split(X, y, test_size=test_split)
class NNN(object):
    """N-layered neural network"""
    def __init__(self, inputs, weights, outputs, alpha):
        self.inputs = inputs
        self.outputs = outputs
        self._ALPHA = alpha
        self._num_of_weights = len(weights)
        self._LAYER_DEFS = {}
        self.WEIGHT_DATA = {}
        self.BIAS_DATA = {}
        self.LAYER_FUNC = {}
        self.LAYERS = {}
        for i in range(self._num_of_weights):
            #(in, out, nonlin)
            self._LAYER_DEFS[i] = {'in': weights[i][0],
                                   'out': weights[i][1],
                                   'nonlin': weights[i][2]}
        print(self._LAYER_DEFS)
        self._init_layers()
    
    def _init_layers(self):
        for i in range(self._num_of_weights):
            _in = self._LAYER_DEFS[i]['in']
            _out = self._LAYER_DEFS[i]['out']
            _nonlin = self._LAYER_DEFS[i]['nonlin']
            self.WEIGHT_DATA[i] = np.random.randn(_in, _out)
            self.BIAS_DATA[i] = np.full((1,_out), 1.0)
            self.LAYER_FUNC[i] = _nonlin
    
    def forward_layer(self, prev_layer, next_layer, bias_values, nonlin):
        """Does the actual calcs between layers :)"""
        ret = nonlin(np.dot(prev_layer, next_layer)+bias_values)
        return ret

    def backward_layer(self, w, z, act, prev_delta):
        w_error = prev_delta.dot(w.T)
        cur_delta = w_error * act(z, deriv=True)
        return cur_delta

    def backprop_sgd(self, w, z, act, prev_delta, bias):
        cur_delta = self.backward_layer(w, z, act, prev_delta)
        w += self._ALPHA * z.T.dot(prev_delta)
        bias += -self._ALPHA * np.sum(np.asarray(prev_delta), axis=0, keepdims=True)
        return cur_delta

    def loss_absolute(y, y_hat):
        return y - y_hat

    def fit(self, X_in, Y_in, train_loops=100, batch_size=32, error_metric=None):
        error_history = []
        for j in range(train_loops):
            X_in, Y_in = shuffle(X_in, Y_in)
            x_batches = split_to_minibatch(X_in, batch_size)
            y_batches = split_to_minibatch(Y_in, batch_size)
            epoch_error = 0
            for x, y in zip(x_batches, y_batches):
                # set up layers
                prev_layer = self.LAYERS[0] = x
                for i in range(self._num_of_weights):
                    current_layer = self.forward_layer(prev_layer, 
                                                       self.WEIGHT_DATA[i],
                                                       self.BIAS_DATA[i], 
                                                       self.LAYER_FUNC[i])
                    self.LAYERS[i+1] = current_layer
                    prev_layer = current_layer
                last_layer = current_layer
    
                # calculate errors
                error = y - last_layer
                if error_metric:
                    epoch_error = (epoch_error + error_metric(y, last_layer)) / 2
                else:
                    epoch_error += np.average(abs(error))
                nonlin = self.LAYER_FUNC[self._num_of_weights - 1]
                delta = error * nonlin(last_layer, deriv=True)
    
                prev_delta = delta
                for i in reversed(range(self._num_of_weights)):
                    prev_delta = self.backprop_sgd(self.WEIGHT_DATA[i],
                                                   self.LAYERS[i], 
                                                   self.LAYER_FUNC[i], 
                                                   prev_delta, 
                                                   self.BIAS_DATA[i])
            
            error_history.append(epoch_error)
            if (j % (train_loops/10)) == 0:
                print("loop: {}".format(j))
                print("Guess (rounded): ")
                print(np.round(last_layer[0], 3))
                print("Actual: ")
                print(np.round(y[0], 3))
        return error_history
        
    def evaluate(self, x, y, loss_function):
        # set up layers
        prev_layer = self.LAYERS[0] = x
        for i in range(self._num_of_weights):
            current_layer = self.forward_layer(prev_layer, 
                                               self.WEIGHT_DATA[i],
                                               self.BIAS_DATA[i], 
                                               self.LAYER_FUNC[i])
            self.LAYERS[i+1] = current_layer
            prev_layer = current_layer
        last_layer = current_layer
        return loss_function(y, last_layer)
# load the data
train_df = pd.read_csv('../input/train.csv')


# clean the data
# borrowed from other kagglers: https://www.kaggle.com/hmendonca/testing-engineered-features-lb-1-42
# Find and drop duplicate rows
t = train_df.iloc[:,2:].duplicated(keep=False)
duplicated_indices = t[t].index.values
print("Removed {} duplicated rows: {}".format(len(duplicated_indices), duplicated_indices))
train_df.iat[duplicated_indices[0], 1] = np.expm1(np.log1p(train_df.target.loc[duplicated_indices]).mean()) # keep and update first with log mean
train_df.drop(duplicated_indices[1:], inplace=True) # drop remaining

# Columns to drop because there is no variation in training set
zero_std_cols = train_df.drop("ID", axis=1).columns[train_df.std() == 0]
train_df.drop(zero_std_cols, axis=1, inplace=True)
print("Removed {} constant columns".format(len(zero_std_cols)))


# log transform the data
train_df = train_df.drop(['ID'], axis=1)
train_df = np.log1p(train_df)


# split the data
X = train_df.drop(['target'], axis=1)
Y = train_df[['target']]
# Y = np.array(Y).reshape(len(Y))

# we will use only a subset of the train data so this kaggle kerenel will run quicker...
#from sklearn.model_selection import train_test_split
#X, _, Y, __ = train_test_split(X, Y, test_size=0.33)
Xt, Xv, Yt, Yv = shuffle_split(X, Y)
# define the loss function and evaluate the network
def rmsle(y, y_hat):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y_hat), 2)))

def rmse(y, y_hat):
    return np.sqrt(np.mean(np.power(y-y_hat, 2)))
# define the layers
i_input = Xt.shape[1]
i_out = 1 # regression, single continuos variable

weights = ((i_input, 64, LReLU6),
           (64, 32, LReLU6),
           (32, 16, LReLU6),
           (16, 8, LReLU6),
           (8, i_out, linear))

# init the network
alpha = 0.0001
train_epochs = 30
batch_size = 16

nn = NNN(i_input, weights, i_out, alpha)

# fit the network
error_history = nn.fit(np.array(Xt), np.array(Yt), train_epochs, batch_size, error_metric=rmse)
# evaluate the network
val_loss = nn.evaluate(np.array(Xv), np.array(Yv), rmse)
print('validation loss: {}'.format(val_loss))
def plot_error_history(error_history):
    from matplotlib import pyplot as plt
    epochs = [i for i in range(len(error_history))]
    plt.plot(epochs, error_history)
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE loss')
    plt.grid(True)
    plt.figure(figsize=(10, 6))
    plt.show()

plot_error_history(error_history)