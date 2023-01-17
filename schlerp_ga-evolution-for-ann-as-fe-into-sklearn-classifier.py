import numpy as np
import pandas as pd
import random
import copy
import time
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
#------------------------------------------------------------------------------
# activations
#------------------------------------------------------------------------------
class Activation(object):
    def __init__(self):    
        pass
    def forward(self, x):
        pass
    def __call__(self, x):
        return self.forward(x)

class Linear(Activation):
    def __init__(self, m=1.0, c=0.):
        self.m = m
        self.c = c
    def forward(self, x):
        return (m * x) + c

class ReLU(Activation):
    def __init__(self, stable=True):
        self.stable = stable
    def forward(self, x):
        if self.stable:
            x = np.clip(x, -700, 700)
        return x * (x > 0)

class LeakyReLU(Activation):
    def __init__(self, stable=True, alpha=0.5):
        self.stable = stable
        self.alpha = alpha
    def forward(self, x):
        if self.stable:
            x = np.clip(x, -700, 700)
        return x * (x > 0) + x * self.alpha * (x <= 0)

class Sigmoid(Activation):
    def __init__(self, stable=True):
        self.stable = stable
    def forward(self, x):
        if self.stable:
            x = np.clip(x, -700, 700)
        return 1 / (1 + np.exp(-x))

class Tanh(Activation):
    def __init__(self, stable=True):
        self.stable = stable
    def forward(self, x):
        if self.stable:
            x = np.clip(x, -700, 700)
        return np.tanh(x)
#------------------------------------------------------------------------------
# loss fucntions
#------------------------------------------------------------------------------
def mse(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def neg_accuracy(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)
#------------------------------------------------------------------------------
# kernel initialisers
#------------------------------------------------------------------------------
def he_normal(shape):
    sd = np.sqrt(2./shape[0])
    return np.random.normal(0., sd, shape)

def he_uniform(shape):
    limit = np.sqrt(2./shape[0])
    return np.random.uniform(-limit, limit, shape)

def random_normal(shape, scale=1., mean=0.):
    return np.random.normal(mean, scale, shape)

def init_ones(shape):
    return np.ones(shape)
#------------------------------------------------------------------------------
# NN Layers
#------------------------------------------------------------------------------
class Layer():
    pass

class InputLayer(Layer):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def set_value(self, x):
        if x.shape[1:] != self.input_shape:
            raise Exception(message='data does not match specified input shape!')
        else:
            self.output = x
    
    def call(self):
        return self.output
    
    def get_output_shape(self):
        return self.input_shape[-1]
    
    def __call__(self):
        return self.call()

class DenseLayer(Layer):
    def __init__(self, n_neurons, activation=LeakyReLU(), 
                 with_bias=False, weight_init=he_normal, bias_init=random_normal):
        self.n_neurons = n_neurons
        self.activation = activation
        self.weight_init = weight_init
        self.with_bias = with_bias
        self.bias_init = bias_init
        self.compiled = False
    
    def get_output_shape(self):
        return self.n_neurons
    
    def build(self):
        self.shape = (self.input_layer.get_output_shape(), self.n_neurons)
        self.weights = self.weight_init(self.shape)
        if self.with_bias:
            self.bias = self.bias_init(self.shape[-1])
        self.compiled = True

    def set_weights(self, weights):
        self.weights = weights
    
    def set_bias(self, bias):
        self.bias = bias
        
    def call(self):
        z = np.dot(self.input_layer(), self.weights)
        if self.with_bias:
            z += self.bias
        self.output = self.activation(z)
        return self.output

    def __call__(self, input_layer=None):
        if isinstance(input_layer, Layer):
            self.input_layer = input_layer
            self.build()
            return self
        else:
            if not self.compiled:
                raise Exception('Layer has not been connected yet!')
            return self.call()
#------------------------------------------------------------------------------
# GA classes
#------------------------------------------------------------------------------
class GAIndividual():
    def __init__(self, _input, _output):
        self._input = _input
        self._output = _output
        self.loss = np.inf
        self.sklm = None
    
    def get_layers(self):
        layers = []
        layer = self._output
        while True:
            if isinstance(layer, InputLayer):
                break
            layers.append(layer)
            layer = layer.input_layer
        return layers
    
    def predict(self, X):
        self._input.set_value(X)
        return self._output()


class GeneticPopulationSKLM(object):
    def __init__(self, sklm=RandomForestClassifier, sklm_params={'n_jobs': -1},
                 n_population=50, n_elite=5, top_k=10, mut_rate=0.1):
        self.sklm = sklm
        self.sklm_params = sklm_params
        self.n_population = n_population
        self.n_elite = n_elite
        self.top_k = top_k
        self.mutation_rate = mut_rate

    def do_epoch(self, X, Y, loss, verbose):
        sorted_indivs, population_loss = self.do_scoring(X, Y, loss, verbose)
        next_generation = self.do_breeding(sorted_indivs, verbose)
        return next_generation, population_loss

    def do_scoring(self, X, Y, loss, verbose, test_size=0.5):
        individuals = []
        population_loss = 0
        Xt, Xv, Yt, Yv = train_test_split(X, Y, test_size=test_size)
        for indiv in self.population:
            if indiv.loss == np.inf:
                sklm_features = indiv.predict(Xt)
                indiv.sklm = self.sklm(**self.sklm_params)
                indiv.sklm.fit(sklm_features, Yt)
                y_pred = indiv.sklm.predict(indiv.predict(Xv))
                indiv.loss = loss(Yv, y_pred)
            population_loss += indiv.loss
            individuals.append(indiv)
        sorted_indivs = sorted(individuals, key=lambda x: x.loss)
        return sorted_indivs, population_loss

    def do_breeding(self, sorted_indivs, verbose):
        #if verbose:
            #print('creating next generation...')
        next_generation = sorted_indivs[0:self.n_elite-1]
        #if verbose:
            #print('adding top {} individuals'.format(self.n_elite))
        pop_to_fill = self.n_population - self.n_elite
        #if verbose:
            #print('breeding {} individuals...'.format(pop_to_fill))
        for i in range(pop_to_fill):
            parent1, parent2 = np.random.choice(sorted_indivs[0:self.top_k], 2)
            child = breed(parent1, parent2)
            child = mutate(child)
            next_generation.append(child)
        return next_generation

    def fit(self, X, Y, n_sklm_features=None, n_hidden_layers=3, n_hidden_neurons=10, 
            with_bias=False, n_epochs=100, loss=mse, verbose=True):
        
        n_features = X.shape[1]
        if n_sklm_features == None:
            n_sklm_features = 2 * n_features
        n_classes = Y.shape[1]
        
        if verbose:
            print('creating initial population...')
        self.population = create_population(self.n_population,
                                            n_features,
                                            n_sklm_features,
                                            n_hidden_layers,
                                            n_hidden_neurons,
                                            with_bias)
        loss_history_best = []
        loss_history_avg = []
        for epoch in range(n_epochs):
            start = time.time()
            if verbose:
                print('epoch #{}'.format(epoch))            
            next_generation, population_loss = self.do_epoch(X, Y, 
                                                             loss, verbose)
            avg_pop_loss = population_loss / self.n_population
            if verbose:
                print('finished epoch!')
                print('best loss: {}'.format(next_generation[0].loss))
                print('avg loss: {}'.format(avg_pop_loss))
            self.population = next_generation
            loss_history_best.append(next_generation[0].loss)
            loss_history_avg.append(avg_pop_loss)
            end = time.time()
            print('epoch took {:.2f}s'.format(end-start))
            print('\n')
            
        self.best_individual = self.population[0]
        return loss_history_best, loss_history_avg

    def predict(self, X):
        sklm_features = self.best_individual.predict(X)
        return self.best_individual.sklm.predict(sklm_features)
#------------------------------------------------------------------------------
# GA functions
#------------------------------------------------------------------------------
def breed(parent1, parent2):
    input_layer = copy.copy(parent1._input)
    layer = input_layer
    for layer_1, layer_2 in zip(reversed(parent1.get_layers()), 
                                reversed(parent2.get_layers())):
        breed_mask = np.random.randint(2, size=layer_1.weights.shape)
        
        child_layer = DenseLayer(layer_1.n_neurons, with_bias=layer_1.with_bias, 
                                 activation=copy.copy(layer_1.activation))(layer)
        
        child_layer.set_weights(np.where(breed_mask, 
                                         layer_1.weights, 
                                         layer_2.weights))
        
        if layer_1.with_bias:
            breed_mask = np.random.randint(2, size=layer_1.bias.shape)
            child_layer.set_bias(np.where(breed_mask, 
                                          layer_1.bias, 
                                          layer_2.bias))
        layer = child_layer
        child_layer.compiled = True
    child = GAIndividual(_input=input_layer, _output=layer)
    return child


def mutate(child, mutation_rate=0.01, mutation_passes=10):
    for i in range(mutation_passes):
        if random.random() <= mutation_rate:
            for layer in child.get_layers():
                x_loc = random.choice(range(layer.weights.shape[0]))
                y_loc = random.choice(range(layer.weights.shape[1]))
                layer.weights[x_loc, y_loc] += (random.random() * 2) - 1
                if layer.with_bias:
                    loc = random.choice(range(layer.bias.shape[-1]))
                    layer.bias[loc] += (random.random() * 2) - 1
    return child

def create_population(n_population, n_features, n_classes, 
                      n_hidden_layers, n_hidden_neurons, with_bias=False, 
                      inner_activation=LeakyReLU(), out_activation=Sigmoid()):
    population = []
    for i in range(n_population):
        in_layer = InputLayer(input_shape=(n_features,))
        layer = in_layer
        for i in range(n_hidden_layers):
            hidden = DenseLayer(n_hidden_neurons, with_bias=with_bias, 
                                activation=inner_activation)(layer)
            layer = hidden
        out_layer = DenseLayer(n_classes, with_bias=with_bias, 
                               activation=out_activation)(layer)
        indiv = GAIndividual(_input=in_layer, _output=out_layer)
        population.append(indiv)
    return population
#------------------------------------------------------------------------------
# data utils
#------------------------------------------------------------------------------
def one_hot(x, classes):
    ret = []
    for value in x:
        temp = [0. for _ in range(classes)]
        temp[int(value)-1] = 1.
        ret.append(temp)
    return np.array(ret)
#------------------------------------------------------------------------------
# plotting utils
#------------------------------------------------------------------------------
def graph_loss(loss_best, loss_avg):
    y = loss_best
    y2 = loss_avg
    x = [x for x in range(len(loss_best))]
    min_epoch, min_loss = min(enumerate(loss_best), key=lambda x: x[1])
    plt.xlabel = 'Epochs'
    plt.ylabel = 'error'
    plt.plot(x, y, 'b-', label='Training loss')
    plt.plot(x, y2, 'g-', label='Population avg loss')
    plt.plot(min_epoch, min_loss, 'rx', mew=2, ms=10, label='minimum loss')
    plt.legend()
    plt.show()
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
                'Horizontal_Distance_To_Fire_Points']

X_all_norm, df_mean, df_std = normalise_df(X_all[cols_non_onehot])

# replace columns with normalised versions
X_all = X_all.drop(cols_non_onehot, axis=1)
X_all = pd.concat([X_all_norm, X_all], axis=1)
# split back into test and train sets
X_train = np.array(X_all[:len(X_train)])
X_test = np.array(X_all[len(X_train):])
Y_train = one_hot(Y_train, number_classes)
Xt, Xv, Yt, Yv = train_test_split(X_train, Y_train, test_size=0.20)
print('creating genetic population...')
gp = GeneticPopulationSKLM(n_population=30, n_elite=3,
                           top_k=10, mut_rate=0.9)

print('fitting genetic population...')
loss_history_best, loss_history_avg = gp.fit(Xt, Yt, 
                                             n_hidden_layers=3, n_hidden_neurons=32, 
                                             with_bias=False, 
                                             n_epochs=100, loss=neg_accuracy)
print('predicting using genetic population...')
y_pred = gp.predict(Xv)
print(y_pred.shape)
print(y_pred[0])
print('mse: {}'.format(mse(Yv, y_pred)))
print('mae: {}'.format(mae(Yv, y_pred)))
print('neg acc: {}'.format(neg_accuracy(Yv, y_pred)))
print('acc: {}'.format(1 - neg_accuracy(Yv, y_pred)))
print('plotting loss history...')
graph_loss(loss_history_best, loss_history_avg)
y_pred = gp.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_pred = y_pred.astype(int) + 1

print('max prediction class: {}'.format(np.max(y_pred)))
print('min prediction class: {}'.format(np.min(y_pred)))
sub = pd.DataFrame()
sub['Id'] = ID_test
sub['Cover_Type'] = y_pred
print(sub['Cover_Type'].value_counts())
sub.to_csv('my_submission.csv', index=False)
print('good luck!')
