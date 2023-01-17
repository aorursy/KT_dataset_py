import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
data = pd.read_csv("../input/HR_comma_sep.csv")
data.head()
data.left = data.left.astype(int)
y, X = dmatrices('left~satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+promotion_last_5years+C(sales)+C(salary)', data, return_type='dataframe')
X = np.asmatrix(X)
y = np.ravel(y)
for i in range(1, X.shape[1]):
    xmin = X[:,i].min()
    xmax = X[:,i].max()
    X[:, i] = (X[:, i] - xmin) / (xmax - xmin)
np.random.seed(1)
alpha = 1  # learning rate
beta = np.random.randn(X.shape[1]) # initiate beta randomly
for T in range(500):
    prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))).ravel()  # logistic function
    prob_y = list(zip(prob, y))
    loss = -sum([np.log(p) if y == 1 else np.log(1 - p) for p, y in prob_y]) / len(y) # calculate loss function
    error_rate = 0
    for i in range(len(y)):
        if ((prob[i] > 0.5 and y[i] == 0) or (prob[i] <= 0.5 and y[i] == 1)):
            error_rate += 1;
    error_rate /= len(y)
    if T % 5 ==0 :
        print('T=' + str(T) + ' loss=' + str(loss) + ' error=' + str(error_rate))
    # calculate derivtive
    deriv = np.zeros(X.shape[1])
    for i in range(len(y)):
        deriv += np.asarray(X[i,:]).ravel() * (prob[i] - y[i])
    deriv /= len(y)
    # change value of beta along inverse direction of derivtive
    beta -= alpha * deriv

