import numpy as np 



# 创建特征向量

x = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90])

# 创建标签向量

y = np.array([10, 16, 33, 45, 49, 70, 89, 93, 113, 117])



print('所有数据的特征组成的向量:', x)

print('特征向量的类型:', type(x))

print('特征向量的维度:',x.size)

print('第5个数据的特征（第5只皮卡丘的等级）:',x[4])
import matplotlib.pyplot as plt



# 画散点图，横坐标是皮卡丘等级，纵坐标是皮卡丘攻击。我们调用matplotlib.pyplot库中的scatter函数

plt.scatter(x, y)

# 将图像显示在屏幕上（在jupyter notebook中不调用show函数也可以显示图像，但通常在本地python中运行必须使用show函数，请同学们养成良好习惯，记得使用show函数）

plt.show()
from sklearn import linear_model



# 创建sklearn库中的LinearRegression类的一个实例，命名为linear_regression

linear_regression = linear_model.LinearRegression()

# 调用liner_regression的fit函数。fit函数有两个参数，第一个参数是特征（需要表示成列向量的形式，每一行是一个数据的特征），第二个参数是标签。该函数的作用是用输入的数据得到一个线性函数。

linear_regression.fit(x.reshape([-1,1]), y)#x.reshape([-1,1])是把x变成1×n的二维矩阵



print("sklearn库的线性回归模型得到的线性函数的斜率：", linear_regression.coef_)

print("sklearn库的线性回归模型得到的线性函数的截距：", linear_regression.intercept_)
# 定义线性回归模型：一元线性函数。w和b是两个参数

w = 0.5

b = 1

y_pred = w * x + b

# 设计损失函数：均方误差

mse_loss = np.square(y - y_pred).mean() / 2

print('函数f(x)=0.5x+1的损失函数值是：', mse_loss)
# 进行多少步梯度下降

n_iterations = 5000

# 学习速率

learning_rate = 0.0007



# 参数的初始值

w = 0

b = 0



# 梯度下降

for i in range(n_iterations):

    # 当前函数对标签的预测值

    y_pred = w * x + b

    # 计算梯度

    w_gradient = (x * (y_pred - y)).mean()

    b_gradient = (y_pred - y).mean()

    # 参数更新

    w = w - learning_rate * w_gradient

    b = b - learning_rate * b_gradient

print('用梯度下降方法找到的参数w:', w)

print('用梯度下降方法找到的参数b:', b)
w = 1.3

b = 4.1

y_pred = w * x + b

plt.scatter(x, y)

# 画折线图

plt.plot(x, y_pred, c='r')#c='r'表示画的线的颜色

plt.show()
w = 0

b = 0

for i in range(10):



    y_pred = w * x + b



    mse_loss = np.square(y - y_pred).mean() / 2



    w_gradient = (x * (y_pred - y)).mean()

    b_gradient = (y_pred - y).mean()



    w = w - 0.0008 * w_gradient

    print(w)
# 对特征进行数值缩放

x_scaled = x / 100



n_iterations = 200

learning_rate = 1



w = 0

b = 0



for i in range(n_iterations):

    y_pred = w * x_scaled + b

    mse_loss = np.square(y - y_pred).mean() / 2

    w_gradient = (x_scaled * (y_pred - y)).mean()

    b_gradient = (y_pred - y).mean()

    w = w - learning_rate * w_gradient

    b = b - learning_rate * b_gradient

print('用梯度下降方法找到的参数w:', w)

print('用梯度下降方法找到的参数b:', b)
from sklearn import linear_model



# 特征是两维的，将10个数据的特征表示成一个矩阵

x = np.array(

[

    [11, 9], # 皮卡丘AA的攻击和防御值

    [16, 14],

    [27, 21],

    [38, 35],

    [49, 47],

    [61, 46],

    [89, 54],

    [92, 82],

    [101, 93],

    [117, 90]

]

)

print('所有数据的特征组成的矩阵:\n', x)

print('特征矩阵的维度:',x.shape)

print('第5个数据的特征（第5只皮卡丘的攻击和防御）:',x[4])

y = np.array([20, 29, 44, 64, 82, 110, 112, 139, 158, 176])



linear_regression = linear_model.LinearRegression()

linear_regression.fit(x, y)



print("sklearn库的线性回归模型得到的线性函数的斜率：", linear_regression.coef_)

print("sklearn库的线性回归模型得到的线性函数的截距：", linear_regression.intercept_)
n_iterations = 10000

learning_rate = 0.0002



# 二元线性回归有三个参数，分别是w1，w2和b。我们都将他们初始化为0

w1 = 0

w2 = 0

b = 0



for i in range(n_iterations):

    y_pred = w1 * x[:, 0] + w2 * x[:, 1] + b

    w1_gradient = (x[:, 0] * (y_pred - y)).mean()

    w2_gradient = (x[:, 1] * (y_pred - y)).mean()

    b_gradient = (y_pred - y).mean()

    w1 = w1 - learning_rate * w1_gradient

    w2 = w2 - learning_rate * w2_gradient

    b = b - learning_rate * b_gradient

print('用梯度下降方法找到的参数w1:', w1)

print('用梯度下降方法找到的参数w2:', w2)

print('用梯度下降方法找到的参数b:', b)
n_iterations = 10000

learning_rate = 0.0002



# np.zeros(3)生成一个3个元素全为0的向量。我们把第一个元素当作w1，第二个元素当作w2，第三个元素当作b

weight = np.zeros(3)

# 为每一个数据添加第三维特征，该特征大小都为1.于是我们可以用向量的内积来直接得到线性函数的输出。

# 例如参数向量是weight=[w1,w2,b], 特征向量a=[x1,x2,1]，则w1*x1+w2*x2+b就是向量weight和a的内积。

x_concat = np.hstack([x, np.ones([10, 1])])

# 此时x_concat.shape的输出为(10,3)



for i in range(n_iterations):

    # 计算对所有数据的标签的预测值，得到一个10维的向量y_pred

    y_pred = np.dot(x_concat, weight)

    

    # 写成一行的形式

    # w_gradient = (x_concat * np.tile((y_pred - y).reshape([-1, 1]), 3)).mean(axis=0)

    

    # 接下来我们将这一行代码进行拆解

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

print('用梯度下降方法找到的参数:', weight)