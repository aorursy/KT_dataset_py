# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Reading the csv file into a dataframe

data = pd.read_csv('../input/mines-vs-rocks/sonar.all-data.csv', header=None)
# View the top five rows of the data

data.head()
X = data.drop(60, axis=1)
print(X.shape)

X.head()
Y = data[60]

print(Y.head())

Y.replace({'R':0, 'M':1}, inplace=True)
Y
from sklearn.model_selection import train_test_split

X_trn, X_tst, y_trn, y_tst = train_test_split(X, Y, test_size=0.2, random_state=0)
def sigmoid(x):

    '''

    Returns the sigmoid of x

    '''

    return 1/(1+np.exp(-x))
def init_param(X):

    '''

    Args: X -> input matrix

    Returns: a tuple of initialised parameters , w vector and b for the bias

    '''

    np.random.seed(0)

    w = np.random.rand(X.shape[1])

    b = 0

    param = {'w': w,

             'b': b

            }

    return param

def Cost_func(X, Y, param, lamb):

    '''

    Computes the cost and gradient value for a given X, Y, param

    Args:

        X => input feautre matrix 

        Y => output array

        param => parameters

        lamb => regularisation constant

    Returns: 

        J => Cost 

        grad => gradient values for w and b

    '''

    m = len(X)

    prob = np.dot(param['w'], X.T) + param['b']

    pred = sigmoid(prob)

    

    J = - np.sum((Y* np.log(pred)) + ((1-Y)*(np.log(1-pred))))/m

    reg = lamb*np.sum(np.power(param['w'], 2))/(2*m)

    J = J + reg

    

    dw = (1/m)* np.dot((pred - Y), X) + lamb*(param['w'])/m

    db = (1/m)* np.sum(pred -Y)

    

    grad= {'dw': dw,

           'db': db}

    

    return J, grad
def optimize(X, Y, param, num_iter, learning_rate, lamb):

    '''

    Optimize the paramenter using gradient descent.

    Args:

        X => input feature matrix

        Y => output array

        param => parameters which are updated

        num_iter => total number of itertions

        learning_rate => learning rate to specify the step size during gradient descent

    Returns:

        param => updated parameters' array

        cost => final cost        

    '''

    for i in range(num_iter):

        cost, grad = Cost_func(X, Y, param, lamb)

        param['w'] = param['w']-(learning_rate*grad['dw'])

        param['b'] = param['b']-(learning_rate*grad['db'])

        

        

    return param, cost
def predict(X, param):

    '''

    predicts the target value, Y_pred

    Args: X => input feature matrix,

          param => parameter vector (w, b)

    Returns: Array of predictions

    '''

    Y_pred=[]

    prob= np.dot(param['w'], X.T) + param['b']

    pred = sigmoid(prob)

    for i in range(len(X)):

        if pred[i] > 0.5:

            Y_pred.append(1)

        else:

            Y_pred.append(0)

            

    return np.array(Y_pred).T

        
def accuracy(Y, Y_pred):

    '''

    Returns the accuracy for the predictions

    '''

    return np.sum(Y == Y_pred) / len(Y)
def main(X, Y, learning_rate=0.01, num_iter=10000, lamb=0, ret_cost=False):

    '''

    Combining all the possible

    Args:

        ret_cost => Flag to return cost along with other values

    Returns: 

        if ret_cost == False

        param and y_pred

        

        else:

        param, y_pred and final cost

    '''

    param = init_param(X)

    cost, grad= Cost_func(X, Y, param, lamb)

    param, cost_f = optimize(X, Y, param, num_iter, learning_rate, lamb)

    

    y_pred = predict(X, param)

    acc = accuracy(Y, y_pred)

    

    print('Final_cost \t', cost_f)

    print('Accuracy_train \t', acc)

    if ret_cost:

        return param, y_pred, cost_f

    else:

        return param, y_pred
import matplotlib.pyplot as plt

cost=[]

num=[]

for i in range(2000, 20000, 2000):

    cost.append(main(X_trn, y_trn, learning_rate= 0.01, num_iter=i, lamb=1, ret_cost=True)[2])

    num.append(i)
plt.title('Variation in cost with number of iterations')

plt.xlabel('number of iterations')

plt.ylabel('Cost')

plt.scatter(num, cost)
rate = [0.001, 0.003, 0.005, 0.009, 0.01, 0.03, 0.05, 0.09, 0.1, 0.3, 0.5]

cst=[]
for r in rate:

    print('Learning_rate: ', r)

    cst.append(main(X_trn, y_trn, learning_rate= r, num_iter=18000, lamb=1, ret_cost=True)[2])

    
plt.title('Variation in Cost with Learning rate')

plt.xlabel('Learning rate')

plt.ylabel('Cost')

plt.plot(rate, cst)
lmd = [ 0.01, 0.1, 0.5, 1, 10, 20, 50, 100]

cost_l = []
for l in lmd:

    print('Regularisation Constatnt: ', l)

    trn_val = main(X_trn, y_trn, learning_rate= 0.5, num_iter=18000, lamb=l, ret_cost=True)

    cost_l.append(trn_val[2])

    p = trn_val[0]

    pred=  predict(X_tst, p)

    print('Test Cost : \t', accuracy(y_tst, pred) )
plt.title('Variation in Cost with Regularisation Constant')

plt.xlabel('Lambda')

plt.ylabel('Cost')

plt.plot(lmd, cost_l)
param_f = main(X_trn, y_trn, learning_rate= 0.5, num_iter=18000, lamb=0.5)[0]
pred= predict(X_tst, param_f)
accuracy(y_tst, pred)