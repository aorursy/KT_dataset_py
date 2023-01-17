# handle imports
import copy
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
class Activation(object):
    
    def __init__(self):    
        pass
    
    def forward(self, x):
        pass
    
    def backward(self, x):
        pass
    
    def __call__(self, x, deriv=False):
        if deriv:
            return self.backward(x)
        return self.forward(x)


class Linear(Activation):
    
    def __init__(self, m=1.0, c=0.):
        self.m = m
        self.c = c
    
    def forward(self, x):
        return (m * x) + c
    
    def backward(self, x):
        return np.ones(x.shape) * m


class ReLU(Activation):
    
    def __init__(self, stable=True):
        self.stable = stable
    
    def forward(self, x):
        if self.stable:
            x = np.clip(x, -700, 700)
        return x * (x > 0)
    
    def backward(self, x):
        return 1. * (x > 0)


class LeakyReLU(Activation):
    
    def __init__(self, stable=True, alpha=0.5):
        self.stable = stable
        self.alpha = alpha
    
    def forward(self, x):
        if self.stable:
            x = np.clip(x, -700, 700)
        return x * (x > 0) + x * self.alpha * (x <= 0)
    
    def backward(self, x):
        return 1. * (x > 0) + self.alpha * (x <= 0)


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
def mse(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))
class Network(object):
    def __init__(self, weights, activation=Sigmoid()):
        self.weights = weights
        self.activation = activation
        self.num_layers = len(weights)
        self.loss = np.inf
    
    def predict(self, x):
        layer = x
        for w in self.weights:
            z = self.activation(np.dot(layer, w.T))
            layer = z
        return z

class GeneticPopulationOld(object):
    def __init__(self, n_population=50, n_elite=5, top_k=10, mut_rate=0.1):
        self.n_population = n_population
        self.n_elite = n_elite
        self.top_k = top_k
        self.mutation_rate = mut_rate

    def fit(self, X, Y, n_hidden=10, n_epochs=100, loss=mse, verbose=True, maximize=False):
        n_features = X.shape[1]
        n_classes = Y.shape[1]
        if verbose:
            print('creating initial population...')
        self.population = create_population(self.n_population,
                                            n_features,
                                            n_classes,
                                            n_hidden)
        loss_history = []
        for epoch in range(n_epochs):
            individuals = []
            population_loss = 0
            if verbose:
                print('epoch #{}'.format(epoch))
            for indiv in self.population:
                y_pred = indiv.predict(X)
                indiv.score = loss(Y, y_pred)
                population_loss += indiv.score
                individuals.append(indiv)
            if verbose:
                print('sorting population...')
            sorted_indivs = sorted(individuals, 
                                   key=lambda x: x.score,
                                   reverse=maximize)
            if verbose:
                print('creating next generation...')
            next_generation = sorted_indivs[0:self.n_elite-1]
            if verbose:
                print('adding top {} individuals'.format(self.n_elite))
            pop_to_fill = self.n_population - self.n_elite
            if verbose:
                print('breeding {} individuals...'.format(pop_to_fill))
            for i in range(pop_to_fill):
                parent1, parent2 = np.random.choice(sorted_indivs[0:self.top_k], 
                                                    2)
                child_weights = breed(parent1.weights, parent2.weights)
                child_weights = mutate(child_weights)
                next_generation.append(Network(child_weights))
            self.population = next_generation
            avg_pop_loss = population_loss/self.n_population
            loss_history.append(avg_pop_loss)
            if verbose:
                print('finished epoch, total loss: {}'.format(avg_pop_loss))
        self.best_individual = self.population[0]
        return loss_history
    
    def predict(self, X):
        return self.best_individual.predict(X)
    

class GeneticPopulation(object):
    def __init__(self, n_population=50, n_elite=5, top_k=10, mut_rate=0.1):
        self.n_population = n_population
        self.n_elite = n_elite
        self.top_k = top_k
        self.mutation_rate = mut_rate

    def do_epoch(self, X, Y, loss, verbose):
        sorted_indivs, population_loss = self.do_scoring(X, Y, loss, verbose)
        next_generation = self.do_breeding(sorted_indivs, verbose)
        return next_generation, population_loss

    def do_scoring(self, X, Y, loss, verbose):
        individuals = []
        population_loss = 0
        for indiv in self.population:
            y_pred = indiv.predict(X)
            indiv.score = loss(Y, y_pred)
            population_loss += indiv.score
            individuals.append(indiv)
        if verbose:
            print('sorting population...')
        sorted_indivs = sorted(individuals, key=lambda x: x.score)
        return sorted_indivs, population_loss

    def do_breeding(self, sorted_indivs, verbose):
        if verbose:
            print('creating next generation...')
        next_generation = sorted_indivs[0:self.n_elite-1]
        if verbose:
            print('adding top {} individuals'.format(self.n_elite))
        pop_to_fill = self.n_population - self.n_elite
        if verbose:
            print('breeding {} individuals...'.format(pop_to_fill))
        for i in range(pop_to_fill):
            parent1, parent2 = np.random.choice(sorted_indivs[0:self.top_k], 2)
            child_weights = breed(parent1.weights, parent2.weights)
            child_weights = mutate(child_weights)
            next_generation.append(Network(child_weights))
        return next_generation

    def fit(self, X, Y, n_hidden=10, n_epochs=100, loss=mse, verbose=True):
        n_features = X.shape[1]
        n_classes = Y.shape[1]
        if verbose:
            print('creating initial population...')
        self.population = create_population(self.n_population,
                                            n_features,
                                            n_classes,
                                            n_hidden)
        loss_history = []
        for epoch in range(n_epochs):
            if verbose:
                print('epoch #{}'.format(epoch))            
            next_generation, population_loss = self.do_epoch(X, Y, loss, verbose)
            self.population = next_generation
            avg_pop_loss = population_loss / self.n_population
            loss_history.append(avg_pop_loss)
            if verbose:
                print('finished epoch, total loss: {}'.format(avg_pop_loss))
            
        self.best_individual = self.population[0]
        return loss_history
    
    def predict(self, X):
        return self.best_individual.predict(X)
def breed(weights1, weights2):
    child_weights = []
    for weight1, weight2 in zip(weights1, weights2):
        breed_mask = np.random.randint(2, size=weight1.shape)
        child_weights.append(np.where(breed_mask, weight1, weight2))
    return child_weights

def mutate(weights, mutation_rate=0.01):
    if random.random() <= mutation_rate:
        for i in range(len(weights)):
            x_loc = random.choice(range(weights[i].shape[0]))
            y_loc = random.choice(range(weights[i].shape[1]))
            weights[i][x_loc, y_loc] += (random.random() * 2) - 1
    return weights

def create_weights(n_features, n_classes, n_hidden, _min=-1, _max=1):
    weights = [np.random.uniform(_min, _max, (n_hidden, n_features)),
               np.random.uniform(_min, _max, (n_classes, n_hidden))]
    return weights

def create_population(n_population, n_features, n_classes, n_hidden):
    population = []
    for i in range(n_population):
        weights = create_weights(n_features, n_classes, n_hidden)
        population.append(Network(weights))
    return population
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
print('creating genetic population...')
gp = GeneticPopulation(n_population=25, n_elite=3, top_k=10, mut_rate=0.1)
print('fitting genetic population...')
train_error = gp.fit(Xt, Yt, n_hidden=100, n_epochs=100, loss=mse)
print('predicting using genetic population...')
y_pred = gp.predict(Xv)

print('mse: {}'.format(mse(Yv, y_pred)))
print('mae: {}'.format(mae(Yv, y_pred)))
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
graph_loss(train_error)
min_epoch, min_loss = min(enumerate(train_error), key=lambda loss: loss[1])
print('min loss: {}, was acheived at {} epochs'.format(min_loss, min_epoch))
y_pred = gp.predict(X_test)
y_pred = np.argmax(y_pred, axis=1) + 1
y_pred = y_pred.astype(int)

print('max prediction class: {}'.format(np.max(y_pred)))
print('min prediction class: {}'.format(np.min(y_pred)))
sub = pd.DataFrame()
sub['Id'] = ID_test
sub['Cover_Type'] = y_pred
sub.to_csv('my_submission.csv', index=False)
print('good luck!')
sub.head()
