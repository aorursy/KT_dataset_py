import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import sklearn.linear_model
np.random.seed(0)
# helper function to plot decision boundary
def plot_decision_boundary(model, X, y):
    
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)

# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))
# importing dataset
m = 300
X,Y = datasets.make_moons(m, noise = 0.2)
X = X.T
Y = Y.reshape((1,m))

plt.scatter(X[0,:],X[1,:],c=Y,s=40,cmap = plt.cm.Spectral)
print(X.shape)
print(Y.shape)
# using simple logistic regression
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);

plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")
def layer_sizes(X,Y):
    nx = X.shape[0]
    nh = 4
    ny = Y.shape[0]
    return (nx,nh,ny)
def init_parameters(nx,nh,ny):
    
    W1 = np.random.randn(nh,nx)*0.01
    b1 = np.zeros((nh,1))
    W2 = np.random.randn(ny,nh)*0.01
    b2 = np.zeros((ny,1))
    
    parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2" : W2,
        "b2" : b2
    }
    return parameters
def forward_prop(X,parameters):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.matmul(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.matmul(W2,A1) +b2
    A2 = sigmoid(Z2)
    
    cache = {
        "Z1" : Z1,
        "A1" : A1,
        "Z2" : Z2,
        "A2" : A2
    }
    return cache
def compute_cost(A2,Y):
    
    logprobs = np.multiply(Y,np.log(A2)) + np.multiply((1-Y), np.log(1-A2))
    cost = -np.sum(logprobs)/m
    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. 
    return cost
def back_prop(parameters,cache,X,Y):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 - Y
    dW2 = np.matmul(dZ2,A1.T)/m
    db2 = np.sum(dZ2,axis = 1,keepdims = True)/m
    dZ1 = np.multiply( np.matmul(W2.T,dZ2), (1-np.power(A1,2)) ) 
    dW1 = np.matmul(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis = 1,keepdims = True)/m
    
    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    return grads
def update_parameters(parameters,grads,learning_rate= 1.5):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2" : W2,
        "b2" : b2
    }
    return parameters
def ANN_model(X,Y,iterations,learning_rate):
    np.random.seed(3)
    costs = []
    (nx,nh,ny) = layer_sizes(X,Y)
    parameters = init_parameters(nx,nh,ny)
    
    for i in range(iterations):
        cache = forward_prop(X,parameters)
        cost = compute_cost(cache["A2"],Y)
        grads = back_prop(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate)
        
        if(i % 1000 == 0):
            print("cost after iteration %i : %f" %(i,cost))
            costs.append(cost)
    return parameters, costs, learning_rate

def predict(parameters,X):
    
    cache = forward_prop(X,parameters)
    predictions = (cache["A2"] > 0.5)
    return predictions
parameters, costs, learning_rate = ANN_model(X,Y,iterations = 20000,learning_rate= 1.3)

predictions = predict(parameters, X)
# accuracy
accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
print ("Accuracy for {} hidden units: {} %".format(4, accuracy))

# Plot learning curve (with costs)
costs = np.squeeze(costs)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

