import numpy as np 

import pandas as pd 

from sklearn.preprocessing import StandardScaler

from scipy.optimize import minimize
# Extract the train set and the test set:

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print(train.shape)

print(test.shape)



# Separate samples and labels:

label_train = train.label.values

sample_train = train.drop(["label"], axis = 1).values

sample_test = test.values
# Normalize the data:

sc = StandardScaler()

sample_train_norm = sc.fit_transform(sample_train)

sample_test_norm = sc.transform(sample_test)
# Define the sigmoid function:

def sigmoid(z):

    return 1 / (1 + np.exp(-z))
# Define the cost function:

def cost(theta, X, y, learning_rate):

    theta = np.matrix(theta)

    X = np.matrix(X)

    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))

    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))

    reg = (learning_rate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))

    return np.sum(first - second) / (len(X)) + reg
# Define the GD function:

def gradient(theta, X, y, learning_rate):

    theta = np.matrix(theta)

    X = np.matrix(X)

    y = np.matrix(y)

    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((learning_rate / len(X)) * theta)

    grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)

    

    return np.array(grad).ravel()
def one_vs_all(X, y, num_class, learning_rate):

    rows = X.shape[0]

    features = X.shape[1]

    Theta = np.zeros((num_class, features + 1))

    # Add a one column at the head of X:

    X = np.insert(X, 0, values = np.ones(rows), axis = 1)

    # Iteratively take out a class as 1 and the rest as 0 form a binary classification:

    for i in range(1, num_class + 1):

        theta = np.zeros(features + 1)

        y_i = np.array([1 if label == i else 0 for label in y])

        y_i = np.reshape(y_i, (rows, 1))

        # Minimize:

        fmin = minimize(fun = cost, x0 = theta, args = (X, y_i, learning_rate), method = 'TNC', jac = gradient)

        Theta[i-1,:] = fmin.x

    return Theta

def predict_all(X, Theta):

    rows = X.shape[0]

    params = X.shape[1]

    num_labels = Theta.shape[0]

    # Add a one column at the head of X:

    X = np.insert(X, 0, values = np.ones(rows), axis = 1)

    X = np.matrix(X)

    Theta = np.matrix(Theta)

    # Calculate the probabilities:

    h = sigmoid(X * Theta.T)

    # Select the highest one:

    h_argmax = np.argmax(h, axis = 1)

    h_argmax = h_argmax + 1

    

    return h_argmax
Theta = one_vs_all(sample_train_norm, label_train, 10, 0.1)

y_pred = predict_all(sample_test_norm, Theta)

y_pred_fla = y_pred.reshape(-1,)

print(y_pred_fla)

print(y_pred_fla.shape)

print(np.ravel(y_pred_fla))
print("The logistics result gives us: {}".format(y_pred_fla))

prediction_file = pd.DataFrame({"ImageId": range(1, 28001), "Label": np.ravel(y_pred_fla)})

prediction_file.to_csv("prediction.csv", index = False, header = True)