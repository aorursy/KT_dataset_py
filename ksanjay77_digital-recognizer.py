import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.metrics import classification_report, confusion_matrix

%matplotlib inline
def load_data_multiclass(test_count,flag=False):
    # Read input file and select 100 record for display.
    data = pd.read_csv('../input/train.csv').values
    #data = train.as_matrix()
    m_total = data.shape[0]
    m_test = test_count
    m = m_total - m_test
    #Normalize the train_x
    shuffle_index = np.random.permutation(m_total)
    data = data[shuffle_index,]
    #print(data.shape)
    train_x = (1./255)*(data[:m, 1:])
    val_x = (1./255)*(data[m:, 1:])
    train_x = train_x.T
    val_x = val_x.T
    #train_x[train_x>0] = 1
    #print(train_x[0:1,:])
    train_y = (data[:m, 0:1]).reshape(1,m)
    val_y = (data[m:, 0:1]).reshape(1,m_test)
    if(flag):
        print("train_x.shape:"+ str(train_x.shape))
        print("train_y.shape:" + str(train_y.shape))
        print("val_y.shape:" +str(val_y.shape))
        print("val_x.shape:" + str(val_x.shape))
    digits =10
    
    Y_new = np.eye(digits)[train_y.astype('int32')]
    Y_new = Y_new.T.reshape(digits, m)
    
    val_Y_new = np.eye(digits)[val_y.astype('int32')]
    val_Y_new = val_Y_new.T.reshape(digits, m_test)
    #shuffle_index = np.random.permutation(m)
    #X_train, Y_train = train_x[:, shuffle_index], Y_new[:, shuffle_index]
    return train_x, Y_new, val_x, val_Y_new
    #return train_x, Y_new
#X , Y, X_val, Y_val = load_data_multiclass(1000,True)
def load_test_data(flag=True):
    # Read input file and select 100 record for display.
    data = pd.read_csv('../input/test.csv').values
    #data[data>0] = 1
    
    #data = train.as_matrix()
    m = data.shape[0]
    
    #Normalize the train_x
    test_x = (data[0:, 0:784]).T/255
    if(flag):
        print("test_x.shape:"+ str(test_x.shape))
    digits =10
    
    return test_x
#X = load_test_data()
def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s

def compute_cost(Y, aL):
    m = Y.shape[1]
    cost = -(1./m) * np.sum(np.multiply(Y, np.log(aL)))
    
    return cost

def feed_forward(X, params, keep_prob=0.5):

    cache = {}

    cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]
    cache["A1"] = sigmoid(cache["Z1"])
    #drop out start here
    if keep_prob < 1:
        cache["D1"] = np.random.rand(cache["A1"].shape[0],cache["A1"].shape[1])    # Step 1: initialize matrix D1 = np.random.rand(..., ...)                         
        cache["D1"] = (cache["D1"] < keep_prob)                  # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)                       
        cache["A1"] = cache["A1"] * cache["D1"]                  # Step 3: shut down some neurons of A1                       
        cache["A1"] = cache["A1"]/keep_prob                      # Step 4: scale the value of neurons that haven't been shut down
    #dropout end here
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]
    cache["A2"] = sigmoid(cache["Z2"])
    if keep_prob < 1:
        cache["D2"] = np.random.rand(cache["A2"].shape[0],cache["A2"].shape[1])    # Step 1: initialize matrix D1 = np.random.rand(..., ...)                         
        cache["D2"] = (cache["D2"] < keep_prob)                  # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)                       
        cache["A2"] = cache["A2"] * cache["D2"]                  # Step 3: shut down some neurons of A2                       
        cache["A2"] = cache["A2"]/keep_prob                      # Step 4: scale the value of neurons that haven't been shut down
    #dropout end here
    cache["Z3"] = np.matmul(params["W3"], cache["A2"]) + params["b3"]
    cache["A3"] = sigmoid(cache["Z3"])
    if keep_prob < 1:
        cache["D3"] = np.random.rand(cache["A3"].shape[0],cache["A3"].shape[1])    # Step 1: initialize matrix D1 = np.random.rand(..., ...)                         
        cache["D3"] = (cache["D3"] < keep_prob)                  # Step 2: convert entries of D3 to 0 or 1 (using keep_prob as the threshold)                       
        cache["A3"] = cache["A3"] * cache["D3"]                  # Step 3: shut down some neurons of A3                       
        cache["A3"] = cache["A3"]/keep_prob                      # Step 4: scale the value of neurons that haven't been shut down
    #dropout end here
    cache["Z4"] = np.matmul(params["W4"], cache["A3"]) + params["b4"]
    cache["A4"] = np.exp(cache["Z4"]) / np.sum(np.exp(cache["Z4"]), axis=0)

    return cache

def back_propagate(X, Y, params, cache, keep_prob):

    dZ4 = cache["A4"] - Y
    dW4 = (1./m_batch) * np.matmul(dZ4, cache["A3"].T)
    db4 = (1./m_batch) * np.sum(dZ4, axis=1, keepdims=True)

    dA3 = np.matmul(params["W4"].T, dZ4)
    dZ3 = dA3 * sigmoid(cache["Z3"]) * (1 - sigmoid(cache["Z3"]))
    dW3 = (1./m_batch) * np.matmul(dZ3, cache["A2"].T)
    db3 = (1./m_batch) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.matmul(params["W3"].T, dZ3)
    dZ2 = dA2 * sigmoid(cache["Z2"]) * (1 - sigmoid(cache["Z2"]))
    dW2 = (1./m_batch) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1./m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(params["W2"].T, dZ2)
    dA1 = dA1 * cache["D1"]              # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    dA1 = dA1/keep_prob                 # Step 2: Scale the value of neurons that haven't been shut down

    dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))
    dW1 = (1./m_batch) * np.matmul(dZ1, X.T)
    db1 = (1./m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3,"dW4": dW4, "db4": db4,}

    return grads

X_train , Y_train, X_test, Y_test = load_data_multiclass(8000, True)
#X_train , Y_train = load_data_multiclass(True)
# hyperparameters
n_x = X_train.shape[0]
digits = 10
n_h1 = 512
n_h2 = 512
n_h3 = 512
m = X_train.shape[1]
learning_rate = 5
beta = .9
batch_size = 128
batches = -(-m // batch_size)
keep_prob = 0.8

# initialization
params = { "W1": np.random.randn(n_h1, n_x) * np.sqrt(1. / n_x),
           "b1": np.zeros((n_h1, 1)) ,
           "W2": np.random.randn(n_h2, n_h1) * np.sqrt(1. / n_h1),
           "b2": np.zeros((n_h2, 1)) ,
           "W3": np.random.randn(n_h3, n_h2) * np.sqrt(1. / n_h2),
           "b3": np.zeros((n_h3, 1)),
           "W4": np.random.randn(digits, n_h3) * np.sqrt(1. / n_h3),
           "b4": np.zeros((digits, 1))  }

V_dW1 = np.zeros(params["W1"].shape)
V_db1 = np.zeros(params["b1"].shape)
V_dW2 = np.zeros(params["W2"].shape)
V_db2 = np.zeros(params["b2"].shape)
V_dW3 = np.zeros(params["W3"].shape)
V_db3 = np.zeros(params["b3"].shape)
V_dW4 = np.zeros(params["W4"].shape)
V_db4 = np.zeros(params["b4"].shape)

# train
for i in range(30):

    permutation = np.random.permutation(X_train.shape[1])
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train[:, permutation]

    for j in range(batches):

        begin = j * batch_size
        end = min(begin + batch_size, X_train.shape[1] - 1)
        X = X_train_shuffled[:, begin:end]
        Y = Y_train_shuffled[:, begin:end]
        m_batch = end - begin

        cache = feed_forward(X, params,keep_prob)
        grads = back_propagate(X, Y, params, cache,keep_prob)

        V_dW1 = (beta * V_dW1 + (1. - beta) * grads["dW1"])
        V_db1 = (beta * V_db1 + (1. - beta) * grads["db1"])
        V_dW2 = (beta * V_dW2 + (1. - beta) * grads["dW2"])
        V_db2 = (beta * V_db2 + (1. - beta) * grads["db2"])
        V_dW3 = (beta * V_dW3 + (1. - beta) * grads["dW3"])
        V_db3 = (beta * V_db3 + (1. - beta) * grads["db3"])
        V_dW4 = (beta * V_dW4 + (1. - beta) * grads["dW4"])
        V_db4 = (beta * V_db4 + (1. - beta) * grads["db4"])
        
        params["W1"] = params["W1"] - learning_rate * V_dW1
        params["b1"] = params["b1"] - learning_rate * V_db1
        params["W2"] = params["W2"] - learning_rate * V_dW2
        params["b2"] = params["b2"] - learning_rate * V_db2
        params["W3"] = params["W3"] - learning_rate * V_dW3
        params["b3"] = params["b3"] - learning_rate * V_db3
        params["W4"] = params["W4"] - learning_rate * V_dW4
        params["b4"] = params["b4"] - learning_rate * V_db4

    cache = feed_forward(X_train, params,1.0)
    train_cost = compute_cost(Y_train, cache["A4"])
    cache = feed_forward(X_test, params,keep_prob=1.0)
    test_cost = compute_cost(Y_test, cache["A4"])
    print("Epoch {}: training cost = {}, test cost = {}".format(i+1 ,train_cost, test_cost))
    #print("Epoch {}: training cost = {}".format(i+1 ,train_cost))

print("Done. check for the performace")
cache = feed_forward(X_test, params,1.0)
test_cost = compute_cost(Y_test, cache["A4"])
predictions = np.argmax(cache["A4"], axis=0)
labels = np.argmax(Y_test, axis=0)
#print(classification_report(predictions, labels))    

X_test = load_test_data()
cache = feed_forward(X_test, params,keep_prob=1.0)
predictions = np.argmax(cache["A4"], axis=0)
submission = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": predictions})
submission.to_csv("final.csv", index=False, header=True)


