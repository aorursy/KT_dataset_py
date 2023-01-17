import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import time
def sigmoid(X, weight):

    z = np.dot(X, weight)

    return 1 / (1 + np.exp(-z))



def loss(h, y):

    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()



def gradient_descent(X, h, y):

    return np.dot(X.T, (h - y)) / y.shape[0]



def update_weight_loss(weight, learning_rate, gradient):

    return weight - learning_rate * gradient



def log_likelihood(x, y, weights):

    z = np.dot(x, weights)

    ll = np.sum( y*z - np.log(1 + np.exp(z)) )

    return ll



def gradient_ascent(X, h, y):

    return np.dot(X.T, y - h)



def update_weight_mle(weight, learning_rate, gradient):

    return weight + learning_rate * gradient
data = pd.read_csv("../input/titanic/train_data.csv")

print("Dataset size")

print("Rows {} Columns {}".format(data.shape[0], data.shape[1]))
print("Columns and data types")

pd.DataFrame(data.dtypes).rename(columns = {0:'dtype'})
df = data.copy()
df['class'] = df['Survived'].apply(lambda x : 1 if x == "Yes" else 0)

# features will be saved as X and target will be saved as y

X = df[['Sex','Emb_3']].copy()

X2 = df[['Sex','Emb_3']].copy()

y = df['class'].copy()
start_time = time.time()



num_iter = 100000



intercept = np.ones((X.shape[0], 1)) 

X = np.concatenate((intercept, X), axis=1)

theta = np.zeros(X.shape[1])



for i in range(num_iter):

    h = sigmoid(X, theta)

    gradient = gradient_descent(X, h, y)

    theta = update_weight_loss(theta, 0.1, gradient)

    

print("Training time (Log Reg using Gradient descent):" + str(time.time() - start_time) + " seconds")

print("Learning rate: {}\nIteration: {}".format(0.1, num_iter))
result = sigmoid(X, theta)

result
'''from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(fit_intercept=True, max_iter=100000)

clf.fit(df[['Sex','Emb_3']], y)

print("Training time (sklearn's LogisticRegression module):" + str(time.time() - start_time) + " seconds")

print("Learning rate: {}\nIteration: {}".format(0.1, num_iter))'''