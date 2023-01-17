import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

np.random.seed(3)
%matplotlib inline
############################################# DATASET #########################################################################


#Reading the data
data = pd.read_csv("../input/diabetes.csv")

Y = data.iloc[:, data.columns.get_loc("Outcome")]
# print(Y.head())
X = data.drop("Outcome", axis = 1)
# print(X.head())

#Splitting the data into train and test data
train_X_orig, test_X_orig, train_Y, test_Y = train_test_split(X, Y, test_size = 0.4, shuffle = True)


print(train_X_orig.shape, train_Y.shape)
print(test_X_orig.shape, test_Y.shape)

# print(train_X)

#Changing train_data and test_data into feature matrix
train_X_orig = train_X_orig.as_matrix()
test_X_orig = test_X_orig.as_matrix()
train_Y = train_Y.as_matrix()
test_Y = test_Y.as_matrix()

# print(train_Y)


#Seperating labels from train_data and test_data
train_Y = train_Y.reshape(460,1).T
# print(train_Y.shape)


test_Y = test_Y.reshape(308,1).T 
# print(test_Y.shape)


#Taking transpose and deleting the last column of train_X and test_X
train_X_orig = train_X_orig.T
print(train_X_orig.shape)


test_X_orig = test_X_orig.T
print(test_X_orig.shape)   


######################Normalization of the training and testing dataset #############################
train_X = train_X_orig / np.max(train_X_orig)
test_X = test_X_orig / np.max(test_X_orig)

    

####Sigmoid function #######
def sigmoid(z):
    a = 1 / (1 + np.exp(-z))

    return a


# print(sigmoid(5))



print("Shape of train_X", train_X.shape)
print("Shape of train_Y", train_Y.shape)

X_shape = train_X.shape
Y_shape = train_Y.shape

#no of training examples
m_test = test_X.shape[1]
m_train = train_X.shape[1]

print("Number of training examples = ", m_train)
print("Number of testing examples = ",m_test)
def layer_size(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    
    return (n_x, n_h, n_y)
def init_parameter(n_x, n_h, n_y):
    np.random.seed(2)
    
    W1 = (np.random.randn(n_h, n_x)) * 0.1
#     W1 = (np.random.randn(n_h, n_x)) * np.sqrt(2 / n_x)
    b1 = np.zeros((n_h, 1))
    W2 = (np.random.randn(n_y, n_h)) * 0.1
#     W2 = (np.random.randn(n_y, n_h)) * np.sqrt(2 / n_h)
    b2 = np.zeros((n_y, 1))
    
    para = {"W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2}
    
    return para
def forward_prop(X, parameter):
    
    W1 = parameter["W1"]
    b1 = parameter["b1"]
    W2 = parameter["W2"]
    b2 = parameter["b2"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(Z1,0)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2
    }
    
    return A2, cache
def compute_cost(A2, Y, parameters, lambd):
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
     
    cost = -1 / m_train * np.sum(np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2)))
    
#     l2_regu = (np.sum(np.square(W1)) + np.sum(np.square(W2))) * (lambd / (2*m_train))
    
#     cost = cross_entropy_cost + l2_regu
    
    cost = np.squeeze(cost)

    return cost
def back_prop(parameter, cache, X, Y, lambd):
    
    
    
    W1 = parameter["W1"]
    W2 = parameter["W2"]    
    
    A1 = cache["A1"]
    A2 = cache["A2"]    
    
    dZ2 = A2 - Y
    dW2 = (np.dot(dZ2, A1.T)) / m_train
    db2 = (np.sum(dZ2, axis = 1, keepdims= True)) / m_train
    dZ1 = np.multiply(np.dot(W2.T, dZ2) , (1 - np.power(A1, 2)))
    dW1 = (np.dot(dZ1, X.T)) / m_train 
    db1 = (np.sum(dZ1, axis = 1, keepdims= True)) / m_train
    
    grads = {"dW1":dW1,
             "db1":db1,
             "dW2":dW2,
             "db2":db2           
    }
    
    return grads

def update_parameter(parameter, grads, learning_rate):
    
    W1 = parameter["W1"]
    b1 = parameter["b1"]
    W2 = parameter["W2"]
    b2 = parameter["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - (learning_rate * dW1)
    b1 = b1 - (learning_rate * db1)
    W2 = W2 - (learning_rate * dW2)
    b2 = b2 - (learning_rate * db2)
    
    para = {"W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2}
    
    return para
def final_model(X, Y, n_h, num_iteration = 1500, learning_rate = 0.6, lambd = 0.0, print_cost = True):
    np.random.seed(3)
    
    costs = []
    
    #layer sizes
    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[2]    
    
    #initialize parameters
    parameters = init_parameter(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    #in loop
    
    for i in range(num_iteration):
        
        #forward propagation
        A2, cache = forward_prop(X, parameters)
        
        #compute cost
        cost = compute_cost(A2, Y, parameters, lambd)
        #back prop
        grads = back_prop(parameters, cache, X, Y, lambd)
            
        
        #update parameters
        parameters = update_parameter(parameters, grads, learning_rate)
        
        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    
    return parameters
def predict(parameter, X):
    A2, cache = forward_prop(X, parameter)
    pred = (A2 > 0.5)
    print(pred.shape)
    return pred
p = final_model(train_X, train_Y, 5)
pred = predict(p, train_X)
print ('Accuracy: %d' % float((np.dot(train_Y,pred.T) + np.dot(1-train_Y,1-pred.T))/float(train_Y.size)*100) + '%')


prd=predict(p, test_X)
prd
print ('Accuracy: %d' % float((np.dot(test_Y,prd.T) + np.dot(1-test_Y,1-prd.T))/float(test_Y.size)*100) + '%')
