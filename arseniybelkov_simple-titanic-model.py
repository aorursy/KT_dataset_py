import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv') # training set
test = pd.read_csv('../input/titanic/test.csv') # test set

train
# We'll use "PassengerId" as index (it'll start with 1),
# and drop the "Name" column because it makes no sense for model
train = train.set_index('PassengerId').drop('Name', axis = 1) 
# Change the values in "Sex" column ('male' = 1, 'female' = 0)
train = train.replace({'Sex' : {'male' : 1, 'female' : 0}})
# Change "Embarked" column: 'S' = 0, 'C' = 1, 'Q' = 2
train = train.replace({'Embarked' : {'S' : 0, 'C' : 1, 'Q' : 2}})
# We also may drop the "Ticket", "Fare", and "Cabin" columns 
train = train.drop(['Ticket', 'Fare', 'Cabin'], axis = 1)
train
# It's necessary to implement normalization of "Age" column
mean_age = train.Age.mean()
age_deviation = train.Age.std()
train.Age = (train.Age - mean_age)/(age_deviation + 10**(-8))
# And we will repalce NaN ages with its average value
train.Age = train.Age.fillna(train.Age.mean())
# Alse we need to fill NaN Embarked positions
# I fill with 2, (assume that these passengers were the last who went on the ship)
train.Embarked = train.Embarked.fillna(2)

train
X_train = train.loc[:799, 'Pclass':]
Y_train = train.loc[:799, 'Survived'].to_numpy().reshape((X_train.shape[0], 1))
# ~ 10% for cross-validation set 
X_val = train.loc[800:, 'Pclass':]
Y_val = train.loc[800:, 'Survived'].to_numpy().reshape((X_val.shape[0], 1))
def initialize_parameters_regr(n, null = False):
    
    # Initialize parameters for Logistic Regression model
    # W - weights, b - bias unit, (W.shape == (n, 1))
    # If null == false W and b are gaussian distributed random variable with mean == 0 and variance == 1
    # else W and b are zeros
    # n - number of features
    # return parameters = {'W' : W_values, 'b' : b_value}
    
    parameters = dict()
    
    parameters['W'] = np.random.randn(n, 1)
    parameters['b'] = np.random.randn()
    
    if null is True:
        parameters['W'] = 0 * parameters['W']
        parameters['b'] = 0 * parameters['b']
    
    return parameters
parameters_test = initialize_parameters_regr(2, null = True)
parameters_test
def sigmoid(Z):
    
    return 1 / (1 + np.exp(-Z))
sigmoid(0)
def compute_cost_regr(X, Y, parameters, lambd = 0):
    
    # Compute cost and gradients
    # grads - dictionary with gradients
    # return cost, grads
    
    W = parameters['W']
    b = parameters['b']
    m = X.shape[0]
    cost = 0
    grads = dict()
    
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    
    # Cross-entropy cost function
    cost = -1/m * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) + lambd/(2*m) * np.sum(W**2)
                         
    grads['dW'] = 1/m * np.dot(X.T, A-Y) + lambd/m * W
    grads['db'] = 1/m * np.sum(A - Y)
    
    return cost, grads
def update_parameters_regr(parameters, grads, learning_rate = 0.001):
    
    parameters['W'] = parameters['W'] - learning_rate * grads['dW']
    parameters['b'] = parameters['b'] - learning_rate * grads['db']
    
    return parameters
def model_regr(X, Y, num_iterations = 15000, learning_rate = 0.01, lambd = 0, null = False, plot_cost = False):
    
    (m, n) = X.shape
    parameters = initialize_parameters_regr(n, null)
    costs = list()
    
    for i in range(num_iterations):
        
        cost, grads = compute_cost_regr(X, Y, parameters, lambd)
        if i % 100 == 0:
            costs.append(cost)  
        parameters = update_parameters_regr(parameters, grads, learning_rate)
        
        if i % 1000 == 0:
            print('Cost after {} iterations: {}'.format(i, cost))
    
    if plot_cost is True:
        plt.plot(costs)
        plt.title('Cost with learning rate = {}, lambda = {}, num iter = {}'.format(learning_rate, lambd, num_iterations))
        plt.xlabel('iterations (per 100)')
        plt.ylabel('Cost')
        plt.show()
    
    
    return parameters
def prediction_regr(X, Y, parameters):
    
    W = parameters['W']
    b = parameters['b']
    
    Z_pred = np.dot(X, W) + b
    A_pred = sigmoid(Z_pred)
    Y_pred = np.round(A_pred)
    
    pred = np.mean(Y_pred == Y) * 100
    
    return pred, Y_pred

parameters_regr = model_regr(X_train, Y_train, null = True, plot_cost = True)
pred_train, Y_pred_train = prediction_regr(X_train, Y_train, parameters_regr)
pred_val, Y_pred_val = prediction_regr(X_val, Y_val, parameters_regr)
print('## Linear logistic regression ##\n')
print('ACCURACY ON TRAINING SET   {} %'.format(pred_train))
print('ACCURACY ON CROSS_VALIDATION SET   {} %'.format(pred_val))
parameters_regr
df_train = pd.DataFrame(data = {'Y_Actual_train' : Y_train.squeeze(), 'Y_Pred_train' : Y_pred_train.squeeze()})
df_val = pd.DataFrame(data = {'Y_Actual_val' : Y_val.squeeze(), 'Y_Pred_val' : Y_pred_val.squeeze()})
conf_matrix_train = pd.crosstab(df_train['Y_Actual_train'], df_train['Y_Pred_train'], rownames = ['Actual'], colnames = ['Predicted'])
conf_matrix_val = pd.crosstab(df_val['Y_Actual_val'], df_val['Y_Pred_val'], rownames = ['Actual'], colnames = ['Predicted'])
# Confusion matrix for training set
plt.figure(figsize = (10, 8))
sns.heatmap(data = conf_matrix_train, annot = True, cmap = "YlGnBu")
plt.title('Training set confusion matrix')
plt.show()
# Confusion matrix for cross_validation set
plt.figure(figsize = (10, 8))
sns.heatmap(data = conf_matrix_val, annot = True, cmap = "YlGnBu")
plt.title('Cross-validation set confusion matrix')
plt.show()
test
# Drop the "Name" column because it makes no sense for model
test = test.drop('Name', axis = 1) 
# Change the values in "Sex" column ('male' = 1, 'female' = 0)
test = test.replace({'Sex' : {'male' : 1, 'female' : 0}})
# Change "Embarked" column: 'S' = 0, 'C' = 1, 'Q' = 2
test = test.replace({'Embarked' : {'S' : 0, 'C' : 1, 'Q' : 2}})
# We also may drop the "Ticket", "Fare", and "Cabin" columns 
test = test.drop(['Ticket', 'Fare', 'Cabin'], axis = 1)
test
test.Embarked = test.Embarked.fillna(2)
test.Age = (test.Age.fillna(mean_age) - mean_age) / (age_deviation + 10**(-8))
test
pred, Y_subm = prediction_regr(test.loc[:, 'Pclass' : 'Embarked'], np.zeros((418, 1)), parameters_regr)
Y_subm[0:5]
test['Survived'] = Y_subm
test.set_index('PassengerId')['Survived'].astype('int').to_csv('./submission.csv')