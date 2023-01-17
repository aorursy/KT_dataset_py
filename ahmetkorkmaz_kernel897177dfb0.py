import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



#Öncelikle data setini okuyoruz

iris = pd.read_csv("../input/iris.csv")



#Daha sonra türleri string veriden sayısal veriye çeviriyoruz.

iris.loc[iris['species']=='virginica','species']=2

iris.loc[iris['species']=='versicolor','species']=1

iris.loc[iris['species']=='setosa','species'] = 0



#Giriş ve çıkış kolonlarını oluşturuyoruz.

X = iris[['petal_length', 'petal_width','sepal_length','sepal_width']].values.T

Y = iris[['species']].values.T





#Veri setinin neye benzediğini görmek için ilk 5 satırını yazdırıyoruz.

print(iris.head())



#Veri setinin geneli ile ilgili bilgileri yazdırıyoruz.

iris.info()
def initialize_parameters(n_x, n_h, n_y):

    

    np.random.seed(2) # we set up a seed so that our output matches ours although the initialization is random.

    

    W1 = np.random.randn(n_h, n_x) * 0.01 # 1. Katmandaki ağırlık matrisi

    b1 = np.zeros(shape=(n_h, 1))  # 1. katmandaki bias vektörü

    W2 = np.random.randn(n_y, n_h) * 0.01   # 2. Katmandaki Ağırlık matrisi

    b2 = np.zeros(shape=(n_y, 1))  # 2. Katmandaki bias vektörü

       

    #parametreleri dictionary formatında kaydediyoruz 

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2}

    

    return parameters



# Katmanlardaki nöron sayıları belirleniyor.



def layer_sizes(X, Y):

    n_x = X.shape[0] # girişteki değişken sayısı

    n_h = 6 # gizli katmandaki nöron sayısı

    n_y = 3 # çıkış nöron sayısı

    return (n_x, n_h, n_y)
def forward_propagation(X, parameters):



#parametreleri dictionaryden çekiyoruz.     

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    

    

    # Forward Propagation ile A2'yi hesaplıyoruz

    Z1 = np.dot(W1, X) + b1

    A1 = np.tanh(Z1)  #tanh activation aktivasyon fonksiyonu

    Z2 = np.dot(W2, A1) + b2

    A2 = 1/(1+np.exp(-Z2))  #sigmoid aktivasyon fonksiyonu

    

    #parametreleri dictionary formatında kaydediyoruz 

    cache = {"Z1": Z1,

             "A1": A1,

             "Z2": Z2,

             "A2": A2}

    

    return A2, cache
def compute_cost(A2, Y, parameters):

   

    m = Y.shape[1] # number of training examples

    # Compute the cross-entropy cost

    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))

    cost = - np.sum(logprobs) / m

    

    return cost
def backward_propagation(parameters, cache, X, Y):

# Number of training examples

    m = X.shape[1]

    

    # First, retrieve W1 and W2 from the dictionary "parameters".

    W1 = parameters['W1']

    W2 = parameters['W2']

    

        

    # Retrieve A1 and A2 from dictionary "cache".

    A1 = cache['A1']

    A2 = cache['A2']

    

    # Backward propagation: calculate dW1, db1, dW2, db2. 

    dZ2= A2 - Y

    dW2 = (1 / m) * np.dot(dZ2, A1.T)

    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))

    dW1 = (1 / m) * np.dot(dZ1, X.T)

    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,

             "db1": db1,

             "dW2": dW2,

             "db2": db2}

    

    return grads
def update_parameters(parameters, grads, learning_rate=1):

# Retrieve each parameter from the dictionary "parameters"

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    

    # Retrieve each gradient from the dictionary "grads"

    dW1 = grads['dW1']

    db1 = grads['db1']

    dW2 = grads['dW2']

    db2 = grads['db2']

    

    # Update rule for each parameter

    W1 = W1 - learning_rate * dW1

    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2

    b2 = b2 - learning_rate * db2

    

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2}

    

    return parameters
def nn_model(X, Y, num_iterations=10000, print_cost=False):

    

    n_x,n_h,n_y=layer_sizes(X, Y)

    

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".

    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    

    # Loop (gradient descent



    for i in range(0, num_iterations):

         

        # Forward propagation.

        A2, cache = forward_propagation(X, parameters)

        

        # Cost fonksiyonu

        cost = compute_cost(A2, Y, parameters)

 

        # Backpropagation

        grads = backward_propagation(parameters, cache, X, Y)

 

        # Gradient descent algoritması ile parametreleri güncelliyoruz

        parameters = update_parameters(parameters, grads)



        

        # Her 1000 iterasyonda cost u yazdırıyoruz.

        if print_cost and i % 1000 == 0:

            print ("Cost after iteration %i: %f" % (i, cost))

    return parameters,n_h
parameters = nn_model(X,Y ,num_iterations=10000, print_cost=True)