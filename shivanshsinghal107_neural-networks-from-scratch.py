import pandas as pd
import numpy as np
from scipy.optimize import minimize
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
df = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")
df.shape
df_test = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")
inputs_test = np.ones([df_test.shape[0], df_test.shape[1]])
inputs_test[:, 1:] = df_test.drop(['label'], axis = 1)
y_test = np.array(df_test['label']).reshape(-1, 1)
np.random.seed(4)
# number of input, hidden & output units
n_input = df.shape[1] - 1
n_hidden = 100
n_output = len(df['label'].unique())
input_weights = np.random.randn(n_hidden, n_input+1)
hidden_weights = np.random.randn(n_output, n_hidden+1)
print(input_weights.shape, hidden_weights.shape)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
inputs = np.ones([df.shape[0], df.shape[1]])
inputs[:, 1:] = df.drop(['label'], axis = 1)
y = np.array(df['label']).reshape(-1, 1)
def cost_function(initial_weights, n_input, n_hidden, n_output, X, y, lamb):
    input_weights = np.reshape(initial_weights[:n_hidden * (n_input+1)], (n_hidden, n_input+1))
    hidden_weights = np.reshape(initial_weights[(n_hidden * (n_input+1)):], (n_output, n_hidden+1))
    #print(input_weights.shape, hidden_weights.shape)
    
    m = len(y)
    y_mat = to_categorical(y)
    
    a1 = X
    
    z2 = np.dot(a1, input_weights.T)
    a2 = np.ones([z2.shape[0], z2.shape[1]+1])
    a2[:, 1:] = sigmoid(z2)
    
    z3 = np.dot(a2, hidden_weights.T)
    a3 = sigmoid(z3)
    
    J = np.sum(np.dot(y_mat, np.log(a3).T) + np.dot((1-y_mat), np.log(1-a3).T))/(-m) + (np.sum(input_weights[:, 1:]**2) + np.sum(hidden_weights[:, 1:]**2)) * (lamb/(2*m))
    
    d3 = a3 - y_mat
    d2 = np.dot(d3, hidden_weights[:, 1:]) * sigmoid(z2) * (1 - sigmoid(z2))
    
    delta1 = np.dot(d2.T, a1)
    delta2 = np.dot(d3.T, a2)
    
    input_weights_grad = delta1 / m
    input_weights_grad[:, 1:] += (input_weights[:, 1:] * lamb) / m
    hidden_weights_grad = delta2 / m
    hidden_weights_grad[:, 1:] += (hidden_weights[:, 1:] * lamb) / m
    #print(input_weights_grad.shape, hidden_weights_grad.shape)
    
    grads = np.concatenate([input_weights_grad.ravel(), hidden_weights_grad.ravel()])
    #print(grads.shape)
    #print(J)
    return J, grads
def cal_accuracy(inputs, y, best_in_weights, best_hid_weights):
    h1 = sigmoid(np.dot(inputs, best_in_weights.T))
    pred1 = np.ones([h1.shape[0], h1.shape[1]+1])
    pred1[:, 1:] = h1
    pred2 = sigmoid(np.dot(pred1, best_hid_weights.T))
    preds = []
    for i in range(pred2.shape[0]):
        arr = list(pred2[i, :])
        idx = arr.index(max(arr))
        preds.append(idx)
    return accuracy_score(y, preds)
initial_weights = np.concatenate([input_weights.ravel(), hidden_weights.ravel()], axis = 0)
print(initial_weights.shape)
lambda_ = 1
costfunction = lambda w: cost_function(w, n_input, n_hidden, n_output, inputs, y, lambda_)
#options = {"maxiter": 200}
res = minimize(fun = costfunction, x0 = initial_weights, jac = True, method = 'TNC')#, options = options)
print(res)
final_weights = res.x
input_weights_ = np.reshape(final_weights[:n_hidden * (n_input+1)], (n_hidden, n_input+1))
hidden_weights_ = np.reshape(final_weights[(n_hidden * (n_input+1)):], (n_output, n_hidden+1))
print(input_weights_.shape, hidden_weights_.shape)
print(f"train accuracy: {cal_accuracy(inputs, y, input_weights_, hidden_weights_)}")
print(f"test accuracy: {cal_accuracy(inputs_test[:3000, :], y_test[:3000, :], input_weights_, hidden_weights_)}")
def gradient_descent(initial_weights, n_input, n_hidden, n_output, X, y, alpha, lamb, epochs):
    input_weights = np.reshape(initial_weights[:n_hidden * (n_input+1)], (n_hidden, n_input+1))
    hidden_weights = np.reshape(initial_weights[(n_hidden * (n_input+1)):], (n_output, n_hidden+1))
    #print(input_weights.shape, hidden_weights.shape)
    
    m = len(y)
    y_mat = to_categorical(y)
    
    a1 = X
    
    for i in range(epochs):
        z2 = np.dot(a1, input_weights.T)
        a2 = np.ones([z2.shape[0], z2.shape[1]+1])
        a2[:, 1:] = sigmoid(z2)

        z3 = np.dot(a2, hidden_weights.T)
        a3 = sigmoid(z3)

        J = np.sum(np.dot(y_mat, np.log(a3).T) + np.dot((1-y_mat), np.log(1-a3).T))/(-m) + (np.sum(input_weights[:, 1:]**2) + np.sum(hidden_weights[:, 1:]**2)) * (lamb/(2*m))

        d3 = a3 - y_mat
        d2 = np.dot(d3, hidden_weights[:, 1:]) * sigmoid(z2) * (1 - sigmoid(z2))

        delta1 = np.dot(d2.T, a1)
        delta2 = np.dot(d3.T, a2)

        input_weights_grad = delta1 / m
        input_weights_grad[:, 1:] += (input_weights[:, 1:] * lamb) / m
        hidden_weights_grad = delta2 / m
        hidden_weights_grad[:, 1:] += (hidden_weights[:, 1:] * lamb) / m
        input_weights -= alpha * input_weights_grad
        hidden_weights -= alpha * hidden_weights_grad
    
    return J, input_weights, hidden_weights
n_hidden = 500
alpha = [0.01, 0.1, 1, 10]
lambda_ = [0.1, 1, 10, 100]
for a in alpha:
    for l in lambda_:
        np.random.seed(4)
        input_weights = np.random.randn(n_hidden, n_input+1)
        hidden_weights = np.random.randn(n_output, n_hidden+1)
        initial_weights = np.concatenate([input_weights.ravel(), hidden_weights.ravel()], axis = 0)

        print(f"alpha: {a}, lambda: {l}")
        J_min, best_in_weights, best_hid_weights = gradient_descent(initial_weights, n_input, n_hidden, n_output,
                                                                    inputs, y, a, l, 10)
        
        print(f"epochs: 10")
        print(f"train accuracy: {cal_accuracy(inputs, y, best_in_weights, best_hid_weights)}")
        print(f"test accuracy: {cal_accuracy(inputs_test[:3000, :], y_test[:3000, :], best_in_weights, best_hid_weights)}")
        print()
def gradient_descent_for_getting_epoch(initial_weights, n_input, n_hidden, n_output, X, y, alpha, lamb, epochs):
    input_weights = np.reshape(initial_weights[:n_hidden * (n_input+1)], (n_hidden, n_input+1))
    hidden_weights = np.reshape(initial_weights[(n_hidden * (n_input+1)):], (n_output, n_hidden+1))
    #print(input_weights.shape, hidden_weights.shape)
    
    m = len(y)
    y_mat = to_categorical(y)
    
    a1 = X
    
    epoch_list = [e for e in range(0, 2001, 100)]
    ins = []
    hids = []
    accs = []
    J_mins = []
    
    for i in range(epochs):
        z2 = np.dot(a1, input_weights.T)
        a2 = np.ones([z2.shape[0], z2.shape[1]+1])
        a2[:, 1:] = sigmoid(z2)

        z3 = np.dot(a2, hidden_weights.T)
        a3 = sigmoid(z3)

        J = np.sum(np.dot(y_mat, np.log(a3).T) + np.dot((1-y_mat), np.log(1-a3).T))/(-m) + (np.sum(input_weights[:, 1:]**2) + np.sum(hidden_weights[:, 1:]**2)) * (lamb/(2*m))

        d3 = a3 - y_mat
        d2 = np.dot(d3, hidden_weights[:, 1:]) * sigmoid(z2) * (1 - sigmoid(z2))

        delta1 = np.dot(d2.T, a1)
        delta2 = np.dot(d3.T, a2)

        input_weights_grad = delta1 / m
        input_weights_grad[:, 1:] += (input_weights[:, 1:] * lamb) / m
        hidden_weights_grad = delta2 / m
        hidden_weights_grad[:, 1:] += (hidden_weights[:, 1:] * lamb) / m
        input_weights -= alpha * input_weights_grad
        hidden_weights -= alpha * hidden_weights_grad
        if i+1 in epoch_list:
            train_acc = cal_accuracy(inputs, y, input_weights, hidden_weights)
            test_acc = cal_accuracy(inputs_test[:3000, :], y_test[:3000, :], input_weights, hidden_weights)
            ins.append(input_weights)
            hids.append(hidden_weights)
            accs.append(test_acc)
            J_mins.append(J)
            print(f"epochs: {i+1}")
            print(f"train accuracy: {train_acc}")
            print(f"test accuracy: {test_acc}")
            
    idx = accs.index(max(accs))
    return J_mins[idx], ins[idx], hids[idx]
np.random.seed(4)
input_weights = np.random.randn(n_hidden, n_input+1)
hidden_weights = np.random.randn(n_output, n_hidden+1)
initial_weights = np.concatenate([input_weights.ravel(), hidden_weights.ravel()], axis = 0)
J, best_in_weights, best_hid_weights = gradient_descent_for_getting_epoch(initial_weights, n_input, n_hidden, n_output, inputs, y, 1, 10, 2000)
print(f"train accuracy: {cal_accuracy(inputs, y, best_in_weights, best_hid_weights)}")
print(f"test accuracy: {cal_accuracy(inputs_test[:3000, :], y_test[:3000, :], best_in_weights, best_hid_weights)}")