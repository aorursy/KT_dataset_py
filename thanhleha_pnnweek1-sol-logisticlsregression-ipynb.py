import numpy as np
import math
data_path = "../input/mnist_seven.csv"
data = np.genfromtxt(data_path, delimiter=",", dtype="uint8")
train, dev, test = data[:4000], data[4000:4500], data[4500:]
def normalize(dataset):
    X = dataset[:, 1:] / 255.     # Normalize input features
    Y = (dataset[:, 0] == 7) * 1  # Convert labels from 0-9 to Is7 (1) or IsNot7(0)
    return X.T,Y.reshape(1, -1)
X_train, Y_train = normalize(train)
print(X_train.shape)
print(Y_train.shape)

X_test, Y_test = normalize(test)
print(X_test.shape)
print(Y_test.shape)

# shuffle the training data since we do SGD
# we shuffle outside the training 
# since we want to compare unvectorized and vectorized versions
# It doesn't affect to batch training later
np.random.seed(8888)     # Do not change those seedings to make our results comparable
np.random.shuffle(train) 

def train_unvectorized(X_train, Y_train, lr=0.2, lambdar=0.0001, epochs=5):
    
    n = X_train.shape[0]
    m = X_train.shape[1]
    
    # Xavier Initialization
    np.random.seed(1234)
    w = np.random.randn(n) * (np.sqrt(2. / (n + 1)))
    b = 0

    for epoch in range(epochs):
        L = 0
        for j in range(m):   # Loop over every training instance
            # Forward pass
            z = 0 
            r = 0 # calculate the regularizer
            for i in range(n):
                z += w[i] * X_train[i,j]
                r += w[i]**2
            z += b
            o = 1. / (1 + math.exp(-z))

            # Calculate the loss
            c = (Y_train[:,j] - o)**2
            l = c / 2 + (lambdar / 2) * r
            L += l

            # Backward pass and update the weights/bias
            for i in range(n):
                w[i] -= lr * (o - Y_train[:,j]) * o * (1 - o) * X_train[i,j] + lambdar * w[i]
            b -= lr * (o - Y_train[:,j]) * o * (1 - o)
        L /= m
        print("Error of the epoch {0}: {1}".format(epoch + 1, L))
    
    return w, b
        
def test_unvectorized(X_test, Y_test, w, b):
    
    n_test = X_test.shape[0]
    m_test = X_test.shape[1]
    corrects = 0
    
    for j in range(m_test):

        # Forward pass
        z = 0 
        for i in range(n_test):
            z += w[i] * X_test[i,j]
        z += b
        o = 1. / (1 + np.exp(-z))
        # Evaluate the outputs
        if ((o >= 0.5) and (Y_test[0, j] == 1)) \
        or ((o < 0.5) and (Y_test[0, j] == 0)):
            corrects +=1
    
    print("Accuracy of our LLSR:" + str((corrects * 100.) / m_test) + "%")
    
    return corrects

w, b = train_unvectorized(X_train, Y_train)
_ = test_unvectorized(X_test, Y_test, w, b)
def train_vectorized(X_train, Y_train, lr=0.2, lambdar=0.0001, epochs=5):
    
    n = X_train.shape[0]
    m = X_train.shape[1]
    
    # Xavier Initialization
    np.random.seed(1234)
    w = np.random.randn(n) * (np.sqrt(2. / (n + 1)))
    b = 0

    for epoch in range(epochs):
        L = 0
        for j in range(m):

            # Forward pass
            z = np.dot(w, X_train[:,j]) + b
            r = np.sum(w**2)
            o = 1. / (1 + np.exp(-z))

            # Calculate the loss (for each instance - SGD) 
            c = (Y_train[:,j] - o)**2
            l = c / 2 + (lambdar / 2) * r
            L += l

            # Backward pass and update the weights/bias (for each instance - SGD) 
            w -= lr * (o - Y_train[:,j]) * o * (1 - o) * X_train[:,j] + lambdar * w 
            b -= lr * (o - Y_train[:,j]) * o * (1 - o)
        L /= m
        print("Error of the epoch {0}: {1}".format(epoch + 1, L))
    return w, b
def test_vectorized(X_test, Y_test, w, b):
    
    m_test = X_test.shape[1]
    result = np.zeros((1, m_test))
    
    z = np.dot(w, X_test) + b
    o = 1. / (1 + np.exp(-z))
    result = (o > 0.5)
    corrects = np.sum(result == Y_test)

    print("Accuracy of our LLSR:" + str((corrects * 100.) / m_test) + "%")
    
    return corrects

w, b = train_vectorized(X_train, Y_train)
_ = test_vectorized(X_test, Y_test, w, b)
def train_batch(X_train, Y_train, lr=0.1, lambdar=0.0001, epochs=50):
    
    n = X_train.shape[0]
    m = X_train.shape[1]

    # Xavier Initialization
    np.random.seed(1234)
    w = np.random.randn(1, n) * (np.sqrt(2. / (n + 1)))
    b = 0

    for epoch in range(epochs):

        # Forward pass
        z = np.dot(w, X_train) + b
        o = 1. / (1 + np.exp(-z))
        r = np.sum(w**2)    

        # Calculate the loss 
        # (axis here makes the training more general \
        # if there are more output neurons than 1, \
        # we want to sum over the instances, so axis=1)
        L = (np.sum((Y_train - o)**2, axis=1) + lambdar * r) / (2 * m)

        # Backward pass and update the weights/bias
        w -= lr * (np.dot((o - Y_train) * o * (1 - o), X_train.T) + lambdar * w) / m
        b -= lr * (np.sum((o - Y_train) * o * (1 - o))) / m

        print("Error of the epoch {0}: {1}".format(epoch + 1, L))
        
    return w, b
        
w_batch, b_batch = train_batch(X_train, Y_train, lr=2, lambdar=0.5, epochs=1001)
_ = test_vectorized(X_test, Y_test, w_batch, b_batch)
w, b = train_vectorized(X_train, Y_train, epochs=1001)
_ = test_vectorized(X_test, Y_test, w, b)
def train_minibatch(X_train, Y_train, batch_size=256, lr=0.1, lambdar=0.0001, epochs=50):
    
    n = X_train.shape[0]
    
    # Xavier Initialization
    np.random.seed(1234)
    w = np.random.randn(1, n) * (np.sqrt(2. / (n + 1)))
    b = 0

    for epoch in range(epochs):
        
        # Split into minibatches into a *list* of sub-arrays
        # we want to split along the number of instances, so axis = 1
        X_minibatch = np.array_split(X_train, batch_size, axis = 1)
        Y_minibatch = np.array_split(Y_train, batch_size, axis = 1) 
        
        # We shuffle the minibatches of X and Y in the same way
        nmb = len(X_minibatch) # number of minibatches
        np.random.seed(5678)
        shuffled_index = np.random.permutation(range(nmb))
                                               
        # Now we can do the training, we cannot vectorize over different minibatches
        # They are like our "epochs"
        for i in range(nmb):
            X_current = X_minibatch[shuffled_index[i]]
            Y_current = Y_minibatch[shuffled_index[i]]
            m = X_current.shape[1]

            # Forward pass
            z = np.dot(w, X_current) + b
            o = 1. / (1 + np.exp(-z))
            r = np.sum(w**2)    

            # Calculate the loss 
            # (axis here makes the training more general \
            # if there are more output neurons than 1, \
            # we want to sum over the instances, so axis=1)
            L = (np.sum((Y_current - o)**2, axis=1) + lambdar * r) / (2 * m)

            # Backward pass and update the weights/bias
            w -= lr * (np.dot((o - Y_current) * o * (1 - o), X_current.T) + lambdar * w) / m
            b -= lr * (np.sum((o - Y_current) * o * (1 - o))) / m

            print("Error of the iteration {0}: {1}".format(epoch * nmb + i + 1, L))

    return w, b
# Do not run this for more than 100 epochs!!!!!!!!!
w_minibatch, b_minibatch = train_minibatch(X_train, Y_train, batch_size=512, lr=0.001, lambdar=0.0001, epochs=35)
_ = test_vectorized(X_test, Y_test, w_minibatch, b_minibatch)