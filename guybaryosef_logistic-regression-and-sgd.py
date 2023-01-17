import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
def processData():
    '''
    Processes the data: Reads input, splits into features and observations, 
    imputes any missing feature data, and deals with categorical features w/ one hot encoding.
    '''
    data = pd.read_csv('../input/nba.games.stats.csv')

    # we will take in some stats and try to predict whether a team lost or won.
    y = data['WINorLOSS']
    new_y = []
    for i in y:
        if i == 'W':
            new_y.append(1)
        else:
            new_y.append(0)
            
    # we wanna base our model off of only our own stats, none of our opponents
    features_to_drop = ['Unnamed: 0', 'Game', 'Date', 'Opponent', 'WINorLOSS', 'OpponentPoints',
                        'Opp.FieldGoals', 'Opp.3PointShotsAttempted', 'Opp.3PointShots.', 'Opp.FreeThrows', 
                        'Opp.FreeThrowsAttempted', 'Opp.FreeThrows.', 'Opp.OffRebounds', 'Opp.TotalRebounds', 
                        'Opp.Assists', 'Opp.Steals', 'Opp.Blocks', 'Opp.Turnovers', 'Opp.TotalFouls',
                        'Opp.FieldGoalsAttempted', 'Opp.FieldGoals.', 'Opp.3PointShots']
    x = data.drop(features_to_drop, axis=1)

    # handle classification data using one-hot encoding
    x = pd.get_dummies(x)
    
    # deal with any possible missing feature data
    feature_imputer = SimpleImputer()
    x = feature_imputer.fit_transform(x)
    
    # normalize and standardize the data off the bat
    norm_x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    # adding columns of 1s to beg. of x, for the y-intercept
    norm_x = np.insert(norm_x, 0, 1, axis=1)
    
    return norm_x, new_y
x_reg, y = processData()
def train_test_val_split(x, y):
    '''
    Split the (x,y) input into 60% training data, 20% validation data, and 20% test data.
    '''
    x_train, x_tmp, y_train, y_tmp = train_test_split(x,y, test_size= 0.4)
    x_test, x_val, y_test, y_val = train_test_split(x_tmp, y_tmp, test_size=0.5)
    return x_train, x_val, x_test, y_train, y_val, y_test
x_train, x_val, x_test, y_train, y_val, y_test = train_test_val_split(x_reg,y)
def logistic_reg_sgd(x_train, y_train, sgd_step, lambd = None, graph = False):
    '''
    Implements logistic regression using stochastic gradient descent as our method of updating our weight vector.
    If a 'lambd' is included, the SGD will be regularized with an L2 penalty. 
    Otherwise, the SGD will be unregularized.
    Inputs are the x and y training data as well as the stochastic gradient descent step size (and an optional lambd).
    If graph is set to true, the function will also keep track of the paramter likelihoods over each iteration.
    Output is the resulting weight vector after doing SGD on all the training data and if graph is set to true than
    also the likelihoods of each iteration.
    '''
    likelihoods = []
    
    w = np.random.normal(0, 1, len(x_train[0])) # initial weight vector
    h = lambda w,x: 1/(1+ np.exp(-(np.transpose(w) @ x))) # sigmoid function
    
    # stochastic gradient descent in action
    if lambd:
        for x,y in zip(x_train, y_train):
            w = w + sgd_step * np.subtract(y, h(w,x)) * x - lambd*w
            
            if graph:
                l = 0
                for i, j in zip(x_train, y_train):
                    l += j*np.log(h(w, i)) + (1-j)*np.log(1-h(w,i))
                likelihoods.append(l)
    else:
        for x,y in zip(x_train, y_train):
            w = w + sgd_step * np.subtract(y, h(w,x)) * x
            
            if graph:
                l = 0
                for i, j in zip(x_train, y_train):
                    l += j*np.log(h(w, i)) + (1-j)*np.log(1-h(w,i))
                likelihoods.append(l)

    if graph:
        return w, likelihoods
    else:
        return w
def find_accuracy(x_test, y_test, weights):
    '''
    Given some test data and the weights of the function, will find the mean squared error of the function.
    '''
    tmp_pred = x_test @ weights
    h = lambda x: 1/( 1+np.exp(-x) ) # sigmoid function

    predictions = [int(round(h(i))) for i in tmp_pred]

    count = 0
    for y, y_hat in zip(y_test, predictions):
        if y==y_hat:
            count += 1
    return count/len(y_test)
def cross_validation(x_train, y_train, x_val, y_val, step):
    '''
    Finds the optimal learning rate parameter for L2 penalized logistic regression, using
    a training and validation set. Returns the optimal learning rate.
    '''
    max_acc = -sys.maxsize
    optimal_lambd = 0
    for i in np.linspace(0.005, 5, 500):
        
        cur_weights = logistic_reg_sgd(x_train, y_train, step, i)
        try:
            cur_acc = find_accuracy(x_val, y_val, cur_weights)
        except:
            continue
            
        if (max_acc < cur_acc):
            max_acc = cur_acc
            optimal_lambd = i
            
    return optimal_lambd
sgd_step = 0.05

unreg_weights, unreg_likelihoods = logistic_reg_sgd(x_train, y_train, sgd_step, graph=True)
unreg_acc = find_accuracy(x_test, y_test, unreg_weights)

optimal_lambd = cross_validation(x_train, y_train, x_val, y_val, sgd_step)
reg_weights, reg_likelihoods = logistic_reg_sgd(x_train, y_train, sgd_step, optimal_lambd, graph=True)
reg_acc = find_accuracy(x_test, y_test, reg_weights)
print("Unregularized Logistic Regression MSE: {:.2f}%".format(unreg_acc*100))
print("L2 Regularized Logistic Regression MSE: {:.2f}%".format(reg_acc*100))
def graphLikelihoods(unreg, reg):
    '''
    Plots the likelihoods over SGD iterations for both the unregularized and regularized logistic regressions.
    '''
    plt.plot(range(len(unreg)), unreg, label='Unregularized')
    plt.plot(range(len(reg)), reg, label='L2 Regulatized')
    plt.title('The Likelihoods of the Logistic Regression parameters')
    plt.xlabel('Iterations over training data')
    plt.ylabel('Likelihood')
    plt.legend()
    plt.savefig('likelihoods_graph.pdf')
    plt.show()
graphLikelihoods(unreg_likelihoods, reg_likelihoods)