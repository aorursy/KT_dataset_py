import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
data = pd.read_csv("../input/HR_comma_sep.csv")
data.shape
data.left = data.left.astype(int)
y, X = dmatrices('left~satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+promotion_last_5years+C(sales)+C(salary)', data, return_type='dataframe')
X = np.asmatrix(X)
y = np.ravel(y)
X[:,4]
y
for i in range(1, X.shape[1]):
    xmin = X[:,i].min()
    xmax = X[:,i].max()
    X[:, i] = (X[:, i] - xmin) / (xmax - xmin)
np.random.seed(1)
alpha = 1  # learning rate
beta = np.random.randn(X.shape[1]) # 随机初始化参数beta,19个
beta
for T in range(500):
    prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))).ravel()  # 根据当前beta预测离职的概率
    prob_y = list(zip(prob, y))
    loss = -sum([np.log(p) if y == 1 else np.log(1 - p) for p, y in prob_y]) / len(y) # 计算损失函数的值
    error_rate = 0
    for i in range(len(y)):
        if ((prob[i] > 0.5 and y[i] == 0) or (prob[i] <= 0.5 and y[i] == 1)):
            error_rate += 1;
    error_rate /= len(y)
    if T % 5 ==0 :
        print('T=' + str(T) + ' loss=' + str(loss) + ' error=' + str(error_rate))
    # 计算损失函数关于beta每个分量的导数
    deriv = np.zeros(X.shape[1])
    for i in range(len(y)):
        deriv += np.asarray(X[i,:]).ravel() * (prob[i] - y[i])
    deriv /= len(y)
    # 沿导数相反方向修改beta
    beta -= alpha * deriv
#dF/dbeta0
prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))).ravel()  # 根据当前beta预测离职的概率
prob_y = list(zip(prob, y))
loss = -sum([np.log(p) if y == 1 else np.log(1 - p) for p, y in prob_y]) / len(y) # 计算损失函数的值
print(loss)
deriv = np.zeros(X.shape[1])
for i in range(len(y)):
    deriv += np.asarray(X[i,:]).ravel() * (prob[i] - y[i])
deriv /= len(y)
deriv[0]
# add a little delta
delta = 0.0001
beta [0] += delta
# loss calculation
prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))).ravel()  # 根据当前beta预测离职的概率
prob_y = list(zip(prob, y))
loss2 = -sum([np.log(p) if y == 1 else np.log(1 - p) for p, y in prob_y]) / len(y)
print(loss2)
a1 = (loss2 - loss) / delta  # (F(b0+delta, b1, b2,..., bn) - F(b0,b1,...bn)) / delta
a2 = deriv[0]  # 用公式 算出来的导数
print(a1)
# err deviration =  e.g. 0.0001
abs(a1-a2)/abs(a2)
