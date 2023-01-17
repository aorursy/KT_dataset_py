# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv('/kaggle/input/iris/Iris.csv')
df.head()
df.info()
df.describe()
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
def assign(x):

    if x=='Iris-setosa':

        return 1

    if x=='Iris-versicolor':

        return 2

    if x=='Iris-virginica':

        return 3

    

y_in = df['Species'].apply(assign)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_in, test_size=0.4)
X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.values.reshape(1,y_train.shape[0]), y_test.values.reshape(1,y_test.shape[0])
def layer_sizes(X, y):

    n_x = X.shape[0]

    n_h = 5

    n_y = y.shape[0]

    

    return n_x, n_h, n_y
def initialize_parameters(n_x, n_h, n_y):

    W1 = np.random.randn(n_h, n_x)*0.01

    b1 = np.zeros((n_h, 1))

    W2 = np.random.randn(n_y, n_h)*0.01

    b2 = np.zeros((n_y, 1))

    

    parameters = {'W1':W1,

                 'b1':b1,

                 'W2':W2,

                 'b2':b2}

    

    return parameters
def sigmoid(z):

    s = 1/(1+np.exp(-z))

    return s



def forward_propagation(X, parameters):

    W1=parameters['W1']

    b1=parameters['b1']

    W2=parameters['W2']    

    b2=parameters['b2']

    

    Z1 = np.dot(W1, X)+b1

    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1)+b2

    A2 = Z2

    

    cache = {'Z1':Z1,

            'A1':A1,

            'Z2':Z2,

            'A2':A2}

    

    return A2, cache

    
def compute_cost(A2, Y, parameters):

    m = Y.shape[1]

    

    logprobs = Y*np.log(A2)+(1-Y)*np.log(1-A2)

    cost = -np.sum(logprobs)/m

    

    return cost
def backward_propagation(parameters, cache, X, Y):

    m = X.shape[1]

    

    W1 = parameters['W1']

    W2 = parameters['W2']

    

    A1 = cache['A1']

    A2 = cache['A2']

    

    dZ2 = A2-Y

    dW2 = np.dot(dZ2, A1.T)/m

    db2 = np.sum(dZ2, axis=1, keepdims=True)/m

    dZ1 = np.dot(W2.T, dZ2)*(1-A1**2)

    dW1 = np.dot(dZ1, X.T)/m

    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    

    grads = {'dW1':dW1,

            'db1':db1,

            'dW2':dW2,

            'db2':db2}

    

    return grads
def update_parameters(parameters, grads, alpha=0.009):

    W1=parameters['W1']

    b1=parameters['b1']    

    W2=parameters['W2']

    b2=parameters['b2']

    

    dW1=grads['dW1']

    db1=grads['db1']

    dW2=grads['dW2']

    db2=grads['db2']

    

    W1 = W1-alpha*dW1

    b1 = b1-alpha*db1

    W2 = W2-alpha*dW2

    b2 = b2-alpha*db2

    

    parameters={'W1':W1,

               'b1':b1,

               'W2':W2,

               'b2':b2}

    

    return parameters
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):

    n_x=layer_sizes(X, Y)[0]

    n_y=layer_sizes(X, Y)[2]

    

    parameters = initialize_parameters(n_x, n_h, n_y)

    

#     grdient descent

    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, parameters)

        

        cost = compute_cost(A2, Y, parameters)

        

        grads = backward_propagation(parameters, cache, X, Y)

        

        parameters = update_parameters(parameters, grads)

        

        if print_cost and i%1000==0:

            print(f'Cost after iteration {i} is {cost}')

        

    return parameters
def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)

    predictions = np.round(A2).astype(int)

    

    return predictions
parameters = nn_model(X_train, y_train, n_h=4,

                     num_iterations=50000, print_cost=False)
predictions = predict(parameters, X_test)
from sklearn.metrics import accuracy_score

out = pd.DataFrame({'real':y_test[0], 'predicted':predictions[0]})

score = accuracy_score(out['real'], out['predicted'])

print(score*100)