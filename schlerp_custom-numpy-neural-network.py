# handle imports
import copy
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
# define initialisers
#------------------------------------------------------------------------------

# plain initialisation
def random_normal(shape, scale=1., mean=0.):
    return np.random.normal(mean, scale, shape)

def random_uniform(shape, scale=1.):
    return np.random.uniform(-scale, scale, shape)

def init_zeros(shape):
    return np.zeros(shape)

def init_ones(shape):
    return np.ones(shape)

def init_constant(shape, constant):
    return np.full(shape, constant)


# lecun initialisations
def lecun_normal(shape):
    sd = np.sqrt(3./shape[0])
    return np.random.normal(0., sd, shape)

def lecun_uniform(shape):
    limit = np.sqrt(3./shape[0])
    return np.random.uniform(-limit, limit, shape)


# he initialisations
def he_normal(shape):
    sd = np.sqrt(2./shape[0])
    return np.random.normal(0., sd, shape)

def he_uniform(shape):
    limit = np.sqrt(2./shape[0])
    return np.random.uniform(-limit, limit, shape)


# glorot initialisations
def glorot_normal(shape):
    sd = np.sqrt(2./np.sum(shape))
    return np.random.normal(0., sd, shape)

def glorot_uniform(shape):
    limit = np.sqrt(2./np.sum(shape))
    return np.random.uniform(-limit, limit, shape)

#------------------------------------------------------------------------------
# define activations
#------------------------------------------------------------------------------

'''
Activations
-----------

this module houses all the activation functio classes. the classes themselves
can be used for calculate the activation function or the gradient of the 
activation function.

'''


class Activation(object):
    
    def __init__(self):    
        pass
    
    def forward(self, x):
        '''runs the activation function
        
        this method is used when the calss is called with the `deriv` parameter
        set to `False`. This will compute the activation function.
        
        Parameters
        ----------
        x : np.array
            the array to apply the activation function to
        
        Returns
        -------
        np.array
            the output array after the activation function has been applied.
        '''
        pass
    
    def backward(self, x):
        '''runs the derivative of the activation function
        
        this method is used when the calss is called with the `deriv` parameter
        set to `True`. This will compute the derivative activation function.
        
        Parameters
        ----------
        x : np.array
            the array to apply the activation function to
        
        Returns
        -------
        np.array
            the output array after the derivative of the activation function 
            has been applied.
        '''
        pass
    
    def __call__(self, x, deriv=False):
        '''runs the activation function (or its derivative)
        
        this class is directly callable, each layer should be passed an 
        instance of one of these activation functions.
        
        Parameters
        ----------
        x : np.array
            the array to apply the activation function to
        deriv : boolean
            controls whether the activation function or its derivative will be 
            ran when called, eg. during forward propagation set to `False` and
            during back propagation set to `True`
        
        Returns
        -------
        np.array
            the output array after the activation function or its derivative 
            has been applied.
        '''
        if deriv:
            return self.backward(x)
        return self.forward(x)


class Linear(Activation):
    
    def __init__(self, m=1.0, c=0.):
        self.m = m
        self.c = c
    
    def forward(self, x):
        r'''returns the applied linear function
        
        this method is used when the calss is called with the `deriv` parameter
        set to `False`. This will compute the activation function.
        
        .. math:: 
            f(x) = mx + c
        
        Parameters
        ----------
        x : array
            the array to apply the activation function to
        
        Returns
        -------
        np.array
            the output array after the activation function has been applied.
        '''
        return (m * x) + c
    
    def backward(self, x):
        r'''runs the derivative of the activation function
        
        this method is used when the calss is called with the `deriv` parameter
        set to `True`. This will compute the derivative activation function.
        
        .. math:: 
            f'(x) = m
        
        Parameters
        ----------
        x : array
            the array to apply the activation function to
        
        Returns
        -------
        np.array
            the output array after the derivative of the activation function 
            has been applied.
        '''        
        return np.ones(x.shape) * m


class ReLU(Activation):
    
    def __init__(self, stable=True):
        self.stable = stable
    
    def forward(self, x):
        r'''returns the applied ReLU function
        
        this method is used when the calss is called with the `deriv` parameter
        set to `False`. This will compute the activation function.
        
        The ReLU function is a modern activation fucntion which will return the
        linear function when (`x`) :math:`x>0` and will return `0` when 
        :math:`x<=>0`
        
        .. math:: 
            f(x) =
              \begin{cases}
                x    & \quad \text{if } x>0\\
                0    & \quad \text{if } x<=0
              \end{cases}
        
        Parameters
        ----------
        x : array
            the array to apply the activation function to
        
        Returns
        -------
        np.array
            the output array after the activation function has been applied.
        '''
        if self.stable:
            x = np.clip(x, -700, 700)
        return x * (x > 0)
    
    def backward(self, x):
        r'''runs the derivative of the activation function
        
        this method is used when the calss is called with the `deriv` parameter
        set to `True`. This will compute the derivative activation function.
        
        Due to the fact that the ReLU function will return `x` when :math:`x>0` 
        and `0` when :math:`x<=0`, the gradient is simply `1.0` when 
        :math:`x>0` and `0` when :math:`x<=0`
        
        .. math:: 
            f'(x) =
              \begin{cases}
                1.0  & \quad \text{if } x>0\\
                0    & \quad \text{if } x<=0
              \end{cases}
        
        Parameters
        ----------
        x : array
            the array to apply the activation function to
        
        Returns
        -------
        np.array
            the output array after the derivative of the activation function 
            has been applied.
        '''
        return 1. * (x > 0)


class LeakyReLU(Activation):
    
    def __init__(self, stable=True, alpha=0.5):
        self.stable = stable
        self.alpha = alpha
    
    def forward(self, x):
        r'''returns the applied ReLU function
        
        this method is used when the calss is called with the `deriv` parameter
        set to `False`. This will compute the activation function.
        
        The Leaky ReLU function is an improved version of the ReLU function. It
        is more robust against the dying neuron problem of the ReLU function 
        because it doesnt return a `0` for all values where :math:`x<=0`. 
        Instead it returns a linear function with a very small gradient 
        :math:\alpha. Therefor it can be described as the inear function when 
        ()`x`) :math:`x>0` and will return :math:`\alpha x` when :math:`x<=0`
        
        .. math:: 
            f(x) =
              \begin{cases}
                x & \quad \text{if } x>0\\
                \alpha x  & \quad \text{if } x<=0
              \end{cases}
        
        Parameters
        ----------
        x : array
            the array to apply the activation function to
        
        Returns
        -------
        np.array
            the output array after the activation function has been applied.
        '''        
        if self.stable:
            x = np.clip(x, -700, 700)
        return x * (x > 0) + x * self.alpha * (x <= 0)
    
    def backward(self, x):
        r'''runs the derivative of the activation function
        
        this method is used when the calss is called with the `deriv` parameter
        set to `True`. This will compute the derivative activation function.
        
        Due to the fact that the ReLU function will return `x` when :math:`x>0` 
        and `0` when :math:`x<=>0`, the gradient is simply `1.0` when 
        :math:`x>0` and `0` when :math:`x<=>0`
        
        .. math:: 
            f'(x) =
              \begin{cases}
                1.0  & \quad \text{if } x>0\\
                \alpha & \quad \text{if } x<=0
              \end{cases}
        
        Parameters
        ----------
        x : array
            the array to apply the activation function to
        
        Returns
        -------
        np.array
            the output array after the derivative of the activation function 
            has been applied.
        '''        
        return 1. * (x > 0) + self.alpha * (x <= 0)

class SchlerpReLU(Activation):
    r'''
    .. math::
        
        \alpha_+ &= \text{pos_alpha}\\
        \alpha_- &= \text{neg_alpha}\\
        xstep_+ &= arctanh(\sqrt{1-\alpha_+})\\
        xstep_- &= -arctanh(\sqrt{1-\alpha_-})\\
        ystep_+ &= tanh(xstep_+) - \alpha_+ * xstep_+\\
        ystep_- &= tanh(xstep_-) - \alpha_- * xstep_-
    
    '''
    def __init__(self, stable=True, m=1.0, c=0.0,
                 alpha_pos=0.1, alpha_neg=0.01, 
                 xstep_pos=1.0, xstep_neg=-1.0):
        self.stable = stable
        self.m = m
        self.c = c
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.xstep_pos = xstep_pos
        self.xstep_neg = xstep_neg
        self.ystep_pos = (self.m*self.xstep_pos)+self.c - (self.xstep_pos * self.alpha_pos)
        self.ystep_neg = (self.m*self.xstep_neg)+self.c - (self.xstep_neg * self.alpha_neg)
    
    def forward(self, x):
        r'''returns the applied ReLU function
        
        this method is used when the calss is called with the `deriv` parameter
        set to `False`. This will compute the activation function.
        
        The schlerp ReLU function is an experimental version of the ReLU and
        tanh functions. It can be explained simply as the tanh function, with 
        added linear functions to either end that stop the tanh function from 
        saturating completely since there is no asymptote at `y=-1` and `y=1`.
        
        .. math::
            
            f(x) =
              \begin{cases}
                \alpha x  & \quad \text{if } x<=xstep_-\\
                x         & \quad \text{if } xstep_-<=x<=xstep_+\\
                \alpha x  & \quad \text{if } x<=xstep_+
              \end{cases}
        
        Parameters
        ----------
        x : array
            the array to apply the activation function to
        
        Returns
        -------
        np.array
            the output array after the activation function has been applied.
        '''
        if self.stable:
            x = np.clip(x, -700, 700)
        return ((x * self.alpha_pos) + self.ystep_pos) * (x > self.xstep_pos) + \
               ((self.m*x)+self.c) * (np.logical_and(self.xstep_neg <= x, self.c <= self.xstep_pos)) + \
               ((x * self.alpha_neg) + self.ystep_neg) * (x < self.xstep_neg)
    
    def backward(self, x):
     
        return self.alpha_pos * (x > self.xstep_pos) + \
               self.m * (np.logical_and(self.xstep_neg <= x, self.c <= self.xstep_pos)) + \
               self.alpha_neg * (x < self.xstep_neg)


class SchlerpTanh(Activation):
    r'''
    .. math::
        
        \alpha_+ &= \text{pos_alpha}\\
        \alpha_- &= \text{neg_alpha}\\
        xstep_+ &= arctanh(\sqrt{1-\alpha_+})\\
        xstep_- &= -arctanh(\sqrt{1-\alpha_-})\\
        ystep_+ &= tanh(xstep_+) - \alpha_+ * xstep_+\\
        ystep_- &= tanh(xstep_-) - \alpha_- * xstep_-
    
    '''
    def __init__(self, stable=True, pos_alpha=0.1, neg_alpha=0.01):
        self.stable = stable
        self.pos_alpha = pos_alpha
        self.neg_alpha = neg_alpha
        self.pos_x_step = np.arctanh(np.sqrt(1-self.pos_alpha))
        self.neg_x_step = -np.arctanh(np.sqrt(1-self.neg_alpha))
        self.pos_y_step = np.tanh(self.pos_x_step) - \
                          (self.pos_x_step * self.pos_alpha)
        self.neg_y_step = np.tanh(self.neg_x_step) - \
                          (self.neg_x_step * self.neg_alpha)
    
    def forward(self, x):
        if self.stable:
            x = np.clip(x, -700, 700)
        return ((x * self.pos_alpha) + self.pos_y_step) * (x > self.pos_x_step) + \
               np.tanh(x) * (np.logical_and(self.neg_x_step <= x, x <= self.pos_x_step)) + \
               ((x * self.neg_alpha) + self.neg_y_step) * (x < self.neg_x_step)
    
    def backward(self, x):
        return self.pos_alpha * (x > self.pos_x_step) + \
               (1.0 - np.square(np.tanh(x))) * (np.logical_and(self.neg_x_step <= x, x <= self.pos_x_step)) + \
               self.neg_alpha * (x < self.neg_x_step)


class ELU(Activation):
    
    def __init__(self, stable=True, alpha=1.0):
        self.stable = stable
        self.alpha = alpha
    
    def forward(self, x):
        if self.stable:
            x = np.clip(x, -700, 700)
        return x * (x > 0) + self.alpha * (np.exp(x)-1) * (x <= 0)
    
    def backward(self, x):
        return 1. * (x > 0) + self.alpha * (np.exp(x)-1) * (x <= 0)


class Sigmoid(Activation):
    
    def __init__(self, stable=True):
        self.stable = stable
    
    def forward(self, x):
        if self.stable:
            x = np.clip(x, -700, 700)
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x):
        return np.exp(-x) / np.square(1+np.exp(-x))


class Tanh(Activation):
    
    def __init__(self, stable=True):
        self.stable = stable
    
    def forward(self, x):
        if self.stable:
            x = np.clip(x, -700, 700)
        return np.tanh(x)
    
    def backward(self, x):
        return 1.0 - np.square(np.tanh(x))

#------------------------------------------------------------------------------
# layer functions
#------------------------------------------------------------------------------
'''
helper functions to create layer definitions

a layer definition is just a tuple that contains:
  * input size (number of neurons of previous layer)
  * output size (number of neurons)
  * activation function (just pass in the function, eg. relu not relu() )
'''


def create_input(num_features):
    '''
    creates an input layer

    this function will return a tuple of (None, num_features, None). This is 
    interpreted in aspecial way by the NeuralNetwork class as long as it is 
    the first layer in the layer_defs argument. 

    Basically, it implies how many features the input data will have, and 
    creates a layer definition that can be consumed by the 
    `create_hidden_layer` and `create_output_layer` functions!

    Parameters
    ----------
    num_features : int
        the number of features in the dataset

    Returns
    -------
    tuple
        a tuple consisting of (None, num_features, None)
    '''
    return {'in': None, 'out': num_features, 'act': None}


def create_hidden(previous_layer, neurons, activation):
    '''creates a hidden layer

    this function will return a tuple of (prev_neurons, neurons, activation)
    where `prev_neurons` is the number of neurons in the previous layer, neurons 
    is the number of neurons in this layer.

    Parameters
    ----------
    previous_layer : tuple
        the layer definition of the previous layer
    neurons : int
        the number of neurons in this layer
    activation : function
        the activation function to be used in this layer

    Returns
    -------
    tuple
        a tuple consisting of (prev_neurons, neurons, activation)
    '''
    return {'in': previous_layer['out'], 'out': neurons, 'act': activation}

def create_output(previous_layer, classes, activation):
    '''creates an out layer

    this function will return a tuple of (prev_neurons, classes, activation)
    where `prev_neurons` is the number of neurons in the previous layer, 
    classes is the number of output classes/neurons.

    this function is essentially just the `create_hidden_layer` function with 
    the neurons parameter changed to classes.

    Parameters
    ----------
    previous_layer : tuple
        the layer definition of the previous layer
    classes : int
        the number of neurons in this layer (output classes)
    activation : function
        the activation function to be used in this layer

    Returns
    -------
    tuple
        a tuple consisting of (prev_neurons, classes, activation)
    '''    
    return create_hidden(previous_layer, classes, activation)

#------------------------------------------------------------------------------
# optimizers
#------------------------------------------------------------------------------
def sgd(w, z, act, prev_delta, bias, alpha):
    '''applies SGD/BGD and returns the modified values'''
    cur_delta = backward_layer(w, z, act, prev_delta)
    w += alpha * z.T.dot(prev_delta)
    bias += -alpha * np.sum(prev_delta, axis=0, keepdims=True)
    return cur_delta, w, bias

def momentum(w, z, act, prev_delta, bias, v, alpha, mu):
    '''applies SGD/BGD with momentum and returns the modified values'''
    cur_delta = backward_layer(w, z, act, prev_delta)
    v = mu * v + z.T.dot(prev_delta)
    w += alpha * v
    bias += -alpha * np.sum(prev_delta, axis=0, keepdims=True)
    return cur_delta, w, bias, v

#------------------------------------------------------------------------------
# neural network definition
#------------------------------------------------------------------------------

def forward_layer(z, w, b, act):
    '''Does the forward pass between layers'''
    return act(z.dot(w)+b)


def backward_layer(w, z, act, prev_delta):
    '''does the backwards pass between layers'''
    w_error = prev_delta.dot(w.T)
    return w_error * act(z, deriv=True)


class NeuralNetwork(object):
    def __init__(self, layer_defs):
        self.inputs = layer_defs[0]['out'] # get number of inputs
        self.outputs = layer_defs[-1]['out']
        self.layer_count = len(layer_defs) - 1
        self.momentum_enable = False
        self.layer_defs = layer_defs

    def add_momentum(self, mu=0.9):
        '''add momentum to network (needs to be called before initialise)'''
        self.momentum_mu = mu
        self.momentum_enable = True
        self.velocities = {}
    
    def initialise(self, weight_init=he_uniform, bias_init=init_zeros):
        '''initlise all the variables that will be used in the network'''
        self.weights = {}
        self.biass = {}
        self.activations = {}
        self.layers = {}
        for i in range(self.layer_count):
            _in = self.layer_defs[i+1]['in']
            _out = self.layer_defs[i+1]['out']
            self.weights[i] = weight_init((_in, _out))
            self.biass[i] = bias_init((1, _out))
            self.activations[i] = self.layer_defs[i+1]['act']
            if self.momentum_enable:
                self.velocities[i] = np.zeros(_out)
    
    def fit(self, X, Y, batch_size=32, alpha=0.001, epochs=100, info_freq=10):
        loss_array = []
        best_loss = np.inf
        for j in range(epochs):
            X, Y = shuffle(X, Y)
            x_batches = split_to_minibatch(X, batch_size)
            y_batches = split_to_minibatch(Y, batch_size)
            epoch_loss = 0
            for x, y in zip(x_batches, y_batches):
                # set up layers
                prev_layer = self.layers[0] = x
                for i in range(self.layer_count):
                    current_layer = forward_layer(prev_layer, 
                                                  self.weights[i],
                                                  self.biass[i], 
                                                  self.activations[i])
                    self.layers[i+1] = current_layer
                    prev_layer = current_layer
                last_layer = current_layer
    
                # calculate errors
                error = y - last_layer
                nonlin = self.activations[self.layer_count - 1]
                delta = error * nonlin(last_layer, deriv=True)
    
                epoch_loss += np.square(error).mean()
    
                prev_delta = delta
                for i in reversed(range(self.layer_count)):
                    if self.momentum_enable:
                        (prev_delta,
                        self.weights[i],
                        self.biass[i],
                        self.velocities[i]) = momentum(self.weights[i],
                                                       self.layers[i],
                                                       self.activations[i], 
                                                       prev_delta, 
                                                       self.biass[i],
                                                       self.velocities[i],
                                                       alpha, 
                                                       self.momentum_mu)
                    else:
                        (prev_delta,
                        self.weights[i],
                        self.biass[i]) = sgd(self.weights[i],
                                             self.layers[i], 
                                             self.activations[i], 
                                             prev_delta, 
                                             self.biass[i],
                                             alpha)
            loss_array.append(epoch_loss)
            
            if epoch_loss < best_loss:
                self.best_weights = copy.deepcopy(self.weights)
                self.best_biass = copy.deepcopy(self.biass)
            
            if (j % (epochs/info_freq)) == 0:             
                print("loop: {}".format(j))
                print("Guess (rounded): ")
                print(np.round(last_layer[0], 3))
                print("Actual: ")
                print(np.round(y[0], 3))
        
        return loss_array    

    def evaluate(self, x, y):
        # set up layers
        prev_layer = self.layers[0] = x
        for i in range(self.layer_count):
            current_layer = forward_layer(prev_layer, 
                                          self.best_weights[i],
                                          self.best_biass[i], 
                                          self.activations[i])
            self.layers[i+1] = current_layer
            prev_layer = current_layer
        last_layer = current_layer
        error = np.square(y) - np.square(last_layer)
        total_error = error.mean()
        return total_error
    
    def predict(self, X, batch_size=64):
        preds = []
        x_batches = split_to_minibatch(X, batch_size)
        for x in x_batches:
            # set up layers
            prev_layer = self.layers[0] = x
            for i in range(self.layer_count):
                current_layer = forward_layer(prev_layer, 
                                              self.best_weights[i],
                                              self.best_biass[i], 
                                              self.activations[i])
                self.layers[i+1] = current_layer
                prev_layer = current_layer
            last_layer = current_layer
            for pred in last_layer:
                preds.append(pred)
        return preds
#------------------------------------------------------------------------------
# plotting functions
#------------------------------------------------------------------------------

def graph_loss(loss):
    y = loss
    x = [x for x in range(len(loss))]
    min_epoch, min_loss = min(enumerate(loss), key=lambda x: x[1])
    plt.xlabel = 'Epochs'
    plt.ylabel = 'error'
    plt.plot(x, y, 'b-', label='Training loss')
    plt.plot(min_epoch, min_loss, 'rx', mew=2, ms=20, label='minimum loss')
    plt.legend()
    plt.show()

#------------------------------------------------------------------------------
# data functions
#------------------------------------------------------------------------------

def one_hot(x, classes, zero_based=True):
    '''returns onehot encoded vector for each item in x'''
    ret = []
    for value in x:
        temp = [0. for _ in range(classes)]
        if zero_based:
            temp[int(value)] = 1.
        else:
            temp[int(value)-1] = 1.
        ret.append(temp)
    return np.array(ret)


def split_to_minibatch(z, batch_size=32):
    z_length = len(z)
    current_batch_size = batch_size
    batch_split_points = []
    while current_batch_size < z_length:
        batch_split_points.append(current_batch_size)
        current_batch_size += batch_size
    return np.split(z, batch_split_points)

#------------------------------------------------------------------------------
# set up dataset
number_classes = 7
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# lets take a look...
train_df.head()
# create train datasets
X_train = train_df.drop(['Id', 'Cover_Type'], axis=1)
Y_train = train_df[['Cover_Type']].values
Y_train = Y_train.reshape(len(Y_train))

# create test dataset and ID's
X_test = test_df.drop(['Id'], axis=1)
ID_test = test_df['Id'].values
ID_test = ID_test.reshape(len(ID_test))

# concatenate both together for feature engineering and normalisation
X_all = pd.concat([X_train, X_test], axis=0)
# mean hillshade
def mean_hillshade(df):
    df['mean_hillshade'] = (df['Hillshade_9am'] + df['Hillshade_Noon'] + df['Hillshade_3pm']) / 3
    return df

# calculate the distance to hydrology using pythagoras theorem
def distance_to_hydrology(df):
    df['distance_to_hydrology'] = np.sqrt(np.power(df['Horizontal_Distance_To_Hydrology'], 2) + \
                                          np.power(df['Vertical_Distance_To_Hydrology'], 2))
    return df

# calculate diagnial distance down to sea level?
def diag_to_sealevl(df):
    df['diag_to_sealevel'] = np.divide(df['Elevation'], np.cos(180-df['Slope']))
    return df

# calculate mean distance to features
def mean_dist_to_feature(df):
    df['mean_dist_to_feature'] = (df['Horizontal_Distance_To_Hydrology'] + \
                                  df['Horizontal_Distance_To_Roadways'] + \
                                  df['Horizontal_Distance_To_Fire_Points']) / 3
    return df

X_all = mean_hillshade(X_all)
X_all = distance_to_hydrology(X_all)
X_all = diag_to_sealevl(X_all)
X_all = mean_dist_to_feature(X_all)
# normalise dataset
def normalise_df(df):
    df_mean = df.mean()
    df_std = df.std()    
    df_norm = (df - df_mean) / (df_std)
    return df_norm, df_mean, df_std

# define columsn to normalise
cols_non_onehot = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
                'Horizontal_Distance_To_Fire_Points', 'mean_hillshade',
                'distance_to_hydrology', 'diag_to_sealevel', 'mean_dist_to_feature']

X_all_norm, df_mean, df_std = normalise_df(X_all[cols_non_onehot])

# replace columns with normalised versions
X_all = X_all.drop(cols_non_onehot, axis=1)
X_all = pd.concat([X_all_norm, X_all], axis=1)
# split back into test and train sets
X_train = np.array(X_all[:len(X_train)])
X_test = np.array(X_all[len(X_train):])
Y_train = one_hot(list(Y_train), number_classes, zero_based=False)
Xt, Xv, Yt, Yv = train_test_split(X_train, Y_train, test_size=0.20)
layer_defs = []

# make the input layer
input_layer = create_input(len(Xt[0]))

layer_defs.append(input_layer)

prev_layer = input_layer

# make the hidden layers
num_hidden = 2
num_neurons = 128
for i in range(num_hidden):
    cur_layer = create_hidden(prev_layer, num_neurons, SchlerpTanh())
    layer_defs.append(cur_layer)
    prev_layer = cur_layer

num_hidden = 2
num_neurons = 64
for i in range(num_hidden):
    cur_layer = create_hidden(prev_layer, num_neurons, SchlerpTanh())
    layer_defs.append(cur_layer)
    prev_layer = cur_layer

num_hidden = 2
num_neurons = 32
for i in range(num_hidden):
    cur_layer = create_hidden(prev_layer, num_neurons, SchlerpTanh())
    layer_defs.append(cur_layer)
    prev_layer = cur_layer
    
# add output layer
output_layer = create_output(prev_layer, len(Yt[0]), Sigmoid())
layer_defs.append(output_layer)
# create the neural network
nn = NeuralNetwork(layer_defs)

# add momentum
nn.add_momentum()

# initialise the variables
nn.initialise(weight_init=he_normal, bias_init=init_zeros)
train_error = nn.fit(Xt, Yt, batch_size=64, epochs=500, info_freq=10, alpha=0.0001)
graph_loss(train_error)
min_epoch, min_loss = min(enumerate(train_error), key=lambda loss: loss[1])
print('min loss: {}, was acheived at {} epochs'.format(min_loss, min_epoch))
test_error = nn.evaluate(Xv, Yv)
print(test_error)
y_pred = nn.predict(X_test, batch_size=512)
y_pred = np.argmax(y_pred, axis=1) + 1
y_pred = y_pred.astype(int)

print('max prediction class: {}'.format(np.max(y_pred)))
print('min prediction class: {}'.format(np.min(y_pred)))
sub = pd.DataFrame()
sub['Id'] = ID_test
sub['Cover_Type'] = y_pred
sub.to_csv('my_submission.csv', index=False)
print('good luck!')
