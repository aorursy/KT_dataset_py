import numpy as np



x = np.array(

[

    [72, 52],

    [80, 48],

    [85, 63],

    [77,59],

    [83, 55],

    [50, 100],

    [60, 95],

    [53, 108],

    [47, 103],

    [58, 104]

    

]

)

y = np.array([0,0,0,0,0,1,1,1,1,1])

print('第5只宝可梦的特征:', x[4])
import matplotlib.pyplot as plt



# 杰尼龟散点图

plt.scatter(x[:5,0], x[:5,1])

# 皮卡丘散点图

plt.scatter(x[5:,0], x[5:,1])

plt.show()
from sklearn.linear_model import LogisticRegression



logistic_regression = LogisticRegression()

logistic_regression.fit(x, y)

print("sklearn库的逻辑回归模型得到的斜率：", logistic_regression.coef_)

print("sklearn库的逻辑回归模型得到的截距：", logistic_regression.intercept_)
# 定义sigmoid函数

def sigmoid(z):

    result = 1.0 / (1.0 + np.exp(-1.0 * z))

    return result



n_iterations = 10000

learning_rate = 0.001



weight = np.zeros(3)

x_concat = np.hstack([x, np.ones([10, 1])])



for i in range(n_iterations):

    # 写成一行的形式

    # w_gradient = (x_concat * np.tile((sigmoid(np.dot(x_concat, weight)) - y).reshape([-1, 1]), 3)).mean(axis=0)

    

    # 接下来我们将这一行代码进行拆解

    # 计算对所有数据的标签的预测值，得到一个10维的向量y_pred

    y_pred = sigmoid(np.dot(x_concat, weight))

    # 计算预测误差，得到一个10维的向量error

    error = y_pred - y

    # 转置成列向量，命名为error_reshape

    error_reshape = error.reshape([-1, 1])

    # 将误差复制3次，因为一共有3个参数。误差分别乘以每一维特征得到3个参数的偏导数。error_tile是形状为10*3的矩阵。

    error_tile = np.tile(error_reshape, 3)

    # 计算所有数据的损失函数对每一个参数的偏导数。gradient_all是形状为10*3的矩阵。其中的第i行第j列元素表示第i个数据的预测误差对第j个参数的偏导数。

    gradient_all = x_concat * error_tile

    # 每一个参数的偏导数都对所有数据求平均，也就是每一列求平均。得到一个3维的向量，即梯度。

    w_gradient = gradient_all.mean(axis=0)

    

    weight = weight - learning_rate * w_gradient



print(weight)
# 杰尼龟散点图

plt.scatter(x[:5,0], x[:5,1])

# 皮卡丘散点图

plt.scatter(x[5:,0], x[5:,1])







# 生成一个长度为100的等差数列，起始数是45，终止数是90。x1是一个np.array向量

x1 = np.linspace(45, 90, 100)

# 对应决策面直线的自变量是x1中的元素时，得到因变量为x2中的元素。x2也是一个长度为100的np.array向量

x2 = -(weight[0] * x1 + weight[2]) / weight[1]

plt.plot(x1, x2, c='r')

plt.show()