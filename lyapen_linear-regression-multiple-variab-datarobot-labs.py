import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline
# Loading the data



def load_data():

    from sklearn.datasets import load_boston

    from sklearn.model_selection import train_test_split

    

    boston = load_boston()

    

    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(boston.data, boston.target, test_size=0.33, random_state=42)



    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))

    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    

    return train_set_x.T, train_set_y, test_set_x.T, test_set_y, boston



train_set_x, train_set_y, test_set_x, test_set_y, visualization_set = load_data()
print(train_set_x.shape, train_set_y.shape, test_set_x.shape, test_set_y.shape)
### START CODE HERE ### (≈ 2 lines of code)

m_train = train_set_x.shape[1]

m_test = test_set_x.shape[1]

### END CODE HERE ###



print ("Number of training examples: m_train = " + str(m_train))

print ("Number of testing examples: m_test = " + str(m_test))



print ("\ntrain_set_x shape: " + str(train_set_x.shape))

print ("train_set_y shape: " + str(train_set_y.shape))

print ("test_set_x shape: " + str(test_set_x.shape))

print ("test_set_y shape: " + str(test_set_y.shape))
plt.figure(figsize=(4, 3))

plt.hist(visualization_set.target)

plt.xlabel("Price ($1000s)")

plt.ylabel("Count")

plt.tight_layout()
for index, feature_name in enumerate(visualization_set.feature_names):

    plt.figure(figsize=(4, 3))

    plt.scatter(visualization_set.data[:, index], visualization_set.target)

    plt.ylabel("Price", size=15)

    plt.xlabel(feature_name, size=15)

    plt.tight_layout()
all_set_x = np.concatenate([train_set_x, test_set_x], axis=1)



mean = all_set_x.mean(axis=1, keepdims=True)

std = all_set_x.std(axis=1, keepdims=True)



train_set_x = (train_set_x - mean) / std

test_set_x = (test_set_x - mean) / std
# GRADED FUNCTION: initialize_with_zeros



def initialize_with_zeros(dim):

    """

    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    

    Argument:

    dim -- size of the w vector we want (or number of parameters in this case)

    

    Returns:

    w -- initialized vector of shape (dim, 1)

    b -- initialized scalar (corresponds to the bias)

    """

    

    ### START CODE HERE ### (≈ 2 lines of code)

    w = np.zeros((dim,1), dtype=float)

    b = 0

    ### END CODE HERE ###



    assert(w.shape == (dim, 1))

    assert(isinstance(b, float) or isinstance(b, int))

    

    return w, b
dim = 2

w, b = initialize_with_zeros(dim)

print ("w = " + str(w))

print ("b = " + str(b))
def propagate(w, b, X, Y):

    """

    Implement the cost function and its gradient for the propagation explained above



    Arguments:

    w -- weights, a numpy array of size (number of features, 1)

    b -- bias, a scalar

    X -- data of shape (number of features, number of examples)

    Y -- results of shape (1, number of examples)

    

    Return:

    cost -- cost function for linear regression

    dw -- gradient of the loss with respect to w, thus same shape as w

    db -- gradient of the loss with respect to b, thus same shape as b

    

    Tips:

    - Write your code step by step for the propagation.

    - Use np.dot() to avoid for-loops in favor of code vectorization

    """

    

    m = X.shape[1]

    

    # FORWARD PROPAGATION (FROM X TO COST)

    ### START CODE HERE ### (≈ 2 lines of code)

    H = np.dot(np.transpose(w),X) + b    # compute activation

    cost =1/2/m*np.sum(np.dot(np.transpose(H-Y),(H-Y)))  # compute cost

    ### END CODE HERE ###



    # BACKWARD PROPAGATION (TO FIND GRAD)

    ### START CODE HERE ### (≈ 2 lines of code)

    dw = 1/m*np.dot(X,np.transpose((H-Y)))

    db = 1/m*np.sum(H-Y)

    ### END CODE HERE ###

    

    assert(dw.shape == w.shape)

    assert(db.dtype == float)

    cost = np.squeeze(cost)

    assert(cost.shape == ())

    

    grads = {"dw": dw,

             "db": db}

    

    return grads, cost
w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])

grads, cost = propagate(w, b, X, Y)

print ("dw = " + str(grads["dw"]))

print ("db = " + str(grads["db"]))

print ("cost = " + str(cost))
# GRADED FUNCTION: optimize



def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):

    """

    This function optimizes w and b by running a gradient descent algorithm

    

    Arguments:

    w -- weights, a numpy array of size (number of features, 1)

    b -- bias, a scalar

    X -- data of shape (number of features, number of examples)

    Y -- results of shape (1, number of examples)

    num_iterations -- number of iterations of the optimization loop

    learning_rate -- learning rate of the gradient descent update rule

    print_cost -- True to print the loss every 100 steps

    

    Returns:

    params -- dictionary containing the weights w and bias b

    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function

    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    

    Tips:

    You basically need to write down two steps and iterate through them:

        1) Calculate the cost and the gradient for the current parameters. Use propagate().

        2) Update the parameters using gradient descent rule for w and b.

    """

    

    costs = []

    

    for i in range(num_iterations):

        

        

        # Cost and gradient calculation (≈ 1 line of code)

        ### START CODE HERE ### 

        grads, cost = propagate(w, b, X, Y)

        ### END CODE HERE ###



        # Retrieve derivatives from grads

        dw = grads["dw"]

        db = grads["db"]

        

        # update rule (≈ 2 lines of code)

        ### START CODE HERE ###

        w = w-learning_rate*grads["dw"]

        b = b-learning_rate*grads["db"]

        ### END CODE HERE ###

        

        # Record the costs

        if i % 100 == 0:

            costs.append(cost)

        

        # Print the cost every 100 training iterations

        if print_cost and i % 100 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

    

    params = {"w": w,

              "b": b}

    

    grads = {"dw": dw,

             "db": db}

    

    return params, grads, costs
params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)



print ("w = " + str(params["w"]))

print ("b = " + str(params["b"]))

print ("dw = " + str(grads["dw"]))

print ("db = " + str(grads["db"]))
# GRADED FUNCTION: predict



def predict(w, b, X):

    """

    Predict using learned linear regression parameters (w, b)

    

    Arguments:

    w -- weights, a numpy array of size (number of features, 1)

    b -- bias, a scalar

    X -- data of shape (number of features, number of examples)

    

    Returns:

    H -- a numpy array (vector) containing all predictions for the examples in X

    """

    

    m = X.shape[1]

    

    # Compute vector "H"

    ### START CODE HERE ### (≈ 1 line of code)

    H = np.dot(np.transpose(w),X)+b

    ### END CODE HERE ###

    

    assert(H.shape == (1, m))

    

    return H
w = np.array([[0.1124579],[0.23106775]])

b = -0.3

X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])

print ("predictions = " + str(predict(w, b, X)))
# GRADED FUNCTION: model



def model(X_train, Y_train, X_test, Y_test, num_iterations=3000, learning_rate=0.5, print_cost=False):

    """

    Builds the linear regression model by calling the function you've implemented previously

    

    Arguments:

    X_train -- training set represented by a numpy array of shape (number of features, m_train)

    Y_train -- training values represented by a numpy array (vector) of shape (1, m_train)

    X_test -- test set represented by a numpy array of shape (number of features, m_test)

    Y_test -- test values represented by a numpy array (vector) of shape (1, m_test)

    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters

    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()

    print_cost -- Set to true to print the cost every 100 iterations

    

    Returns:

    d -- dictionary containing information about the model.

    """

    

    ### START CODE HERE ###

    

    # initialize parameters with zeros (≈ 1 line of code)

    w, b = initialize_with_zeros(X_train.shape[0])



    # Gradient descent (≈ 1 line of code)

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=False)

    

    # Retrieve parameters w and b from dictionary "parameters"

    w = parameters["w"]

    b = parameters["b"]

    

    # Predict test/train set examples (≈ 2 lines of code)

    Y_prediction_test = predict(w, b, X_test)

    Y_prediction_train = predict(w, b, X_train)



    ### END CODE HERE ###



    # Print train/test Errors

    print ("Train RMSE: {} ".format(np.sqrt(np.mean((Y_prediction_train - Y_train) ** 2))))

    print ("Test RMSE: {} ".format(np.sqrt(np.mean((Y_prediction_test - Y_test) ** 2))))

    

    d = {"costs": costs,

         "Y_prediction_test": Y_prediction_test, 

         "Y_prediction_train" : Y_prediction_train, 

         "w" : w, 

         "b" : b,

         "learning_rate" : learning_rate,

         "num_iterations": num_iterations}

    

    return d
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=3000, learning_rate=0.05, print_cost=True)
# Training set

plt.figure(figsize=(4, 3))

plt.title("Training set")

plt.scatter(train_set_y, d["Y_prediction_train"])

plt.plot([0, 50], [0, 50], "--k")

plt.axis("tight")

plt.xlabel("True price ($1000s)")

plt.ylabel("Predicted price ($1000s)")

plt.tight_layout()



# Test set

plt.figure(figsize=(4, 3))

plt.title("Test set")

plt.scatter(test_set_y, d["Y_prediction_test"])

plt.plot([0, 50], [0, 50], "--k")

plt.axis("tight")

plt.xlabel("True price ($1000s)")

plt.ylabel("Predicted price ($1000s)")

plt.tight_layout()