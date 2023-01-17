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
            # CODE HERE
            

            # Calculate the loss
            # CODE HERE
            
            
            # Backward pass and update the weights/bias
            # CODE HERE
            pass
        # Accumulate the total loss and print it
        L /= m
        print("Error of the epoch {0}: {1}".format(epoch + 1, L))
    
    return w, b
        
def test_unvectorized(X_test, Y_test, w, b):
    
    n_test = X_test.shape[0]
    m_test = X_test.shape[1]
    corrects = 0
    
    for j in range(m_test):
        
        # Forward pass
        # CODE HERE
        
        # Evaluate the outputs
        # CODE HERE
        
        pass
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
            # CODE HERE
            
            # Calculate the loss (for each instance - SGD) 
            # CODE HERE
            
            # Backward pass and update the weights/bias (for each instance - SGD) 
            # CODE HERE
            pass    
        L /= m
        # print("Error of the epoch {0}: {1}".format(epoch + 1, L))
    return w, b
def test_vectorized(X_test, Y_test, w, b):
    
    m_test = X_test.shape[1]
    corrects = 0
    
    # CODE HERE
    
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
    
    L = 0

    for epoch in range(epochs):

        # Forward pass
        # CODE HERE

        # Calculate the loss 
        # CODE HERE
        
        # Backward pass and update the weights/bias
        # CODE HERE
        pass
        # print("Error of the epoch {0}: {1}".format(epoch + 1, L))
        
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

    L = 0
    for epoch in range(epochs):
        
        # Split into minibatches 
        # CODE HERE
        
        # We shuffle the minibatches of X and Y in the same way
        # CODE HERE
        
        # Now we can do the training, we cannot vectorize over different minibatches
        # They are like our "epochs"
        for i in range(None): # CODE HERE
            
            # Extract a minibatch to do training
            X_current = None # CODE HERE
            Y_current = None # CODE HERE
            m = X_current.shape[1]

            # Forward pass
            # CODE HERE  

            # Calculate the loss 
            # CODE HERE

            # Backward pass and update the weights/bias
            # CODE HERE

            # print("Error of the iteration {0}: {1}".format(None, L)) # CODE HERE

    return w, b
# Do not run this for more than 100 epochs!!!!!!!!!
w_minibatch, b_minibatch = train_minibatch(X_train, Y_train, batch_size=512, lr=0.001, lambdar=0.0001, epochs=30)
_ = test_vectorized(X_test, Y_test, w_minibatch, b_minibatch)
