import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

%matplotlib inline
df = pd.read_csv('../input/voice.csv')

df.head()
np.where(pd.isnull(df))
X = df.drop("label", axis=1)

Y = df["label"]
scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
gender_encoder = LabelEncoder()

Y = gender_encoder.fit_transform(Y)
def normalize_features(features):

    scaler = StandardScaler()

    scaler.fit(features)

    return scaler.transform(features)
def encode_gender(Y):

    gender_encoder = LabelEncoder()

    return gender_encoder.fit_transform(Y)
def read_data(path):

    df = pd.read_csv(path)

    X = df.drop("label", axis=1)

    Y = df["label"]

    X = normalize_features(X)

    Y = encode_gender(Y)

    Y = Y.reshape(Y.shape[0], 1)

    return X,Y
X,Y = read_data('../input/voice.csv')

assert(X.shape == (3168, 20))

assert(Y.shape == (3168, 1))

assert(Y.dtype == np.dtype('int64'))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
def sigmoid(z):

    s = 1 / (1 + np.exp(-z))

    return s
def initialize_weight_and_bias(dimension):

    w = np.zeros((dimension,1))

    b = 0

    return w, b
def forward_prop(X, Y, w, b):

    return sigmoid(np.dot(w.T, X) + b)
def compute_cost(A, Y, m):

    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    return cost
def back_prop(X, A, Y, m):

    dw = (1 / m) * np.dot(X, (A - Y).T)

    db = (1 / m) * np.sum(A - Y)

    return {"dw": dw, "db": db}
def optimize(w, b, X, Y, num_iterations, learning_rate):

    costs = []

    m = X.shape[1]



    for i in range(num_iterations):



        A = forward_prop(X, Y, w, b)

        cost = compute_cost(A, Y, m)

        grads = back_prop(X, A, Y, m)



        dw = grads["dw"]

        db = grads["db"]



        w = w - learning_rate * dw

        b = b - learning_rate * db



        if i % 100 == 0:

            costs.append(cost)





    params = {"w": w,

              "b": b}



    grads = {"dw": dw,

             "db": db}



    return params, grads, costs
def predict(w, b, X):

    m = X.shape[1]

    Y_prediction = np.zeros((1, m))

    w = w.reshape(X.shape[0], 1)



    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):



        if A[0, i] <= 0.5:

            Y_prediction[0, i] = 0

        else:

            Y_prediction[0, i] = 1



    return Y_prediction
def model(X_train, Y_train, X_test, Y_test, num_iterations=100, learning_rate=0.005):



    w, b = initialize_weight_and_bias(X_train.shape[0])



    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)



    w = parameters["w"]

    b = parameters["b"]



    Y_prediction_test = predict(w, b, X_test)

    Y_prediction_train = predict(w, b, X_train)



    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))



    d = {"costs": costs,

         "Y_prediction_test": Y_prediction_test,

         "Y_prediction_train": Y_prediction_train,

         "w": w,

         "b": b,

         "learning_rate": learning_rate,

         "num_iterations": num_iterations}

    return d
result = model(X_train.T, Y_train.T, X_test.T, Y_test.T, num_iterations = 2000, learning_rate = 0.01)