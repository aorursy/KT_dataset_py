import numpy as np

import pandas as pd

from patsy import dmatrices

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

import matplotlib.pyplot as plt
data = pd.read_csv("../input/HR_comma_sep.csv")
data.dtypes
data.head()
pd.crosstab(data.salary, data.left).plot(kind='bar')

plt.show()
q = pd.crosstab(data.salary, data.left)

print(q)
print(q.sum(1)) # add each row
q.div(q.sum(1), axis = 0)
q.div(q.sum(1), axis = 0).plot(kind='bar', stacked = True)
data[data.left == 0].satisfaction_level.hist()

plt.show()
data[data.left == 1].satisfaction_level.hist()

plt.show()
y, X = dmatrices('left~satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+promotion_last_5years+C(sales)+C(salary)', data, return_type='dataframe')
X.head()
y.head()
X = np.asmatrix(X)

y = np.ravel(y)
X[:2]
X.shape
for i in range(1, X.shape[1]): # after dmatrices，first column is "intercept" column,it's all "1"

    xmin = X[:,i].min()

    xmax = X[:,i].max()

    X[:, i] = (X[:, i] - xmin) / (xmax - xmin)
np.random.seed(1) # 保证每次运行结果一致

alpha = 1  # learning rate

beta = np.random.randn(X.shape[1]) # 随机初始化参数beta, 然后 在下面的训练中 不断的拟合。 randn表示正态分布

beta
# 下面两行 就是看一下 prob.ravel的效果，ravel 不能省略
prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta))))

prob
prob.ravel()


for T in range(200):

    # 1. 表示需要float 

    # numpy.ravel https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.rave.html#numpy.ravel

    # Note: 虽然matmul表示矩阵相乘， 但是由于 beta 是列向量，所以就是点积

    # 

    prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))).ravel()  # 根据当前beta预测离职的概率. matmul: matrix multiply 

    prob_y = list(zip(prob, y))

    loss = -sum([np.log(p) if y == 1 else np.log(1 - p) for p, y in prob_y]) / len(y) # 计算损失函数的值. cross entrophy loss function 

    error_rate = 0

    for i in range(len(y)):

        if ((prob[i] > 0.5 and y[i] == 0) or (prob[i] <= 0.5 and y[i] == 1)):

            error_rate += 1;

    error_rate /= len(y)

    

    if T % 5 ==0 :

        print('T=' + str(T) + ' loss=' + str(loss) + ' error=' + str(error_rate))

    # 计算损失函数关于beta每个分量的导数

    deriv = np.zeros(X.shape[1]) # 每个分量的导数 也就是每个feature. deriv是向量

    

    # 对应logistic regression 课件2： Logistic Regression梯度计算 page 里的求偏导的公式

    # 其实质，每个sample data 都要贡献 deriv. prob[i] 其实就是上面根据公式 1 / (1 + e ^ (-betaX))计算出来的概率

    for i in range(len(y)):

        deriv += np.asarray(X[i,:]).ravel() * (prob[i] - y[i])

    deriv /= len(y)

    # 沿导数相反方向修改beta

    beta -= alpha * deriv
Xtrain,Xvali,ytrain,yvali=train_test_split(X, y, test_size=0.2, random_state=3)
np.random.seed(1)

alpha = 5 # learning rate

beta = np.random.randn(Xtrain.shape[1]) # 随机初始化参数beta

error_rates_train=[]

error_rates_vali=[]

for T in range(200):

    prob = np.array(1. / (1 + np.exp(-np.matmul(Xtrain, beta)))).ravel()  # 根据当前beta预测离职的概率

    prob_y = list(zip(prob, ytrain))

    loss = -sum([np.log(p) if y == 1 else np.log(1 - p) for p, y in prob_y]) / len(ytrain) # 计算损失函数的值

    error_rate = 0

    for i in range(len(ytrain)):

        if ((prob[i] > 0.5 and ytrain[i] == 0) or (prob[i] <= 0.5 and ytrain[i] == 1)):

            error_rate += 1;

    error_rate /= len(ytrain)

    error_rates_train.append(error_rate)

    

    prob_vali = np.array(1. / (1 + np.exp(-np.matmul(Xvali, beta)))).ravel()  # 根据当前beta预测离职的概率

    prob_y_vali = list(zip(prob_vali, yvali))

    loss_vali = -sum([np.log(p) if y == 1 else np.log(1 - p) for p, y in prob_y_vali]) / len(yvali) # 计算损失函数的值

    error_rate_vali = 0

    for i in range(len(yvali)):

        if ((prob_vali[i] > 0.5 and yvali[i] == 0) or (prob_vali[i] <= 0.5 and yvali[i] == 1)):

            error_rate_vali += 1

    error_rate_vali /= len(yvali)

    error_rates_vali.append(error_rate_vali)

    

    if T % 5 ==0 :

        print('T=' + str(T) + ' loss=' + str(loss) + ' error=' + str(error_rate)+ ' error_vali=' + str(error_rate_vali))

    

    # Note: 只在train set 上计算导数， 而不需要在valid set 上面计算导数，而计算出来的 新的beta value 将会被用在both train and valid上面 （反向传播修改参数旨在training上）

    # 计算损失函数关于beta每个分量的导数

    deriv = np.zeros(Xtrain.shape[1])

    for i in range(len(ytrain)):

        deriv += np.asarray(Xtrain[i,:]).ravel() * (prob[i] - ytrain[i])

    deriv /= len(ytrain)

    # 沿导数相反方向修改beta

    beta -= alpha * deriv
plt.plot(range(50,200), error_rates_train[50:], 'r^', range(50, 200), error_rates_vali[50:], 'bs')

plt.show()
np.random.seed(1)

alpha = 1  # learning rate

beta = np.random.randn(X.shape[1]) # 随机初始化参数beta



#dF/dbeta0

prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))).ravel()  # 根据当前beta预测离职的概率

prob_y = list(zip(prob, y))

loss = -sum([np.log(p) if y == 1 else np.log(1. - p) for p, y in prob_y]) / len(y) # 计算损失函数的值

deriv = np.zeros(X.shape[1])

for i in range(len(y)):

    deriv += np.asarray(X[i,:]).ravel() * (prob[i] - y[i])

deriv /= len(y)

print('We calculated ' + str(deriv[0]))

 

delta = 0.0001 # 详单与 用导数定义， 求极限的分母部分

beta[0] += delta # 唯一改变的东西，然后下面根据这个改变的的beta 再来计算一次loss 

prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))).ravel()  # 根据当前beta预测离职的概率

prob_y = list(zip(prob, y))

loss2 = -sum([np.log(p) if y == 1 else np.log(1. - p) for p, y in prob_y]) / len(y) # 计算损失函数的值

# 导数的定义来计算deriv 然后比较

shouldbe = (loss2 - loss) / delta # (F(b0+delta,b1,...,bn) - F(b0,...bn)) / delta

print('According to definition of gradient, it is ' + str(shouldbe))