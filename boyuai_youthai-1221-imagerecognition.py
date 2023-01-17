import numpy as np # 导入numpy库
x = np.array([1, 2, 1, 0, 3])

y = np.array([-1, -1, 2, 1, 1])



print(x - y) # 对位相减

print(x + y) # 对位相加

print(x * y) # 对位相乘

print(x @ y) # 点积

print(np.sqrt(x)) # 开根

print(x ** 2) # 每一位平方

print(np.sum(x)) # 求和
np.sqrt(np.sum((x-y)**2)) # 计算欧式距离
(x @ y) / (np.sqrt(np.sum(x**2))) / (np.sqrt(np.sum(y**2))) # 计算cos距离 
A = np.array([

    [1, 0],

    [0, -1],

    [2, 2]

])

B = np.array([

    [0, 1],

    [-1, 1],

    [1, 1]

])

C = np.array([

    [1, 0],

    [2, 1]

])

print(A + B) # 对位相加

print(A * B) # 对位相乘

print(A - B) # 对位相减

print(A @ C) # 矩阵相乘
x = np.array([1, 0, -1, -2]) # 神经网络输入x

w = np.array([

    [1, 0, 0],

    [1, 0, 1],

    [1, 0, 1],

    [1, 0, 0],

])                          # 4×3 的权重矩阵

b = np.array([1, 2, 3])     # 偏置项bias 与输出维度相同

print(w[0]) # 取第一行

print(w[:,0]) # 取第一列，其中:表示该维度全取，所以是每一行都取，取每一行的第一个元素，就是第一列 

w[1:3, 1:3] # 取第二第三行，第二列第三列

w[-1] # 取最后一行

w[-2] # 取倒数第二行



print(x @ w + b) # 全连接层的输出y
x = np.array([1, 0, -1, -2]) 

X = np.array([

    [1, 0],

    [0, -1],

    [2, 2]

])



print(X.flatten()) # 把矩阵X展开成向量

print(np.average(x)) # 对向量x的所有元素取平均

print(np.average(X)) # 对矩阵X的所有元素取平均

print(np.average(X, axis = 1)) # 对矩阵X的每一行取平均

print(np.average(X, axis = 0)) # 对矩阵X的每一列取平均

print(np.zeros(10))

print(np.ones(5))

print(np.random.randn(5, 5)) # 用均值为0，方差为1的高斯分布随机生成5*5的矩阵

print(np.linspace(-5, 5, 101)) # 把-5到5等分成100个区间，因为左闭右闭，所以总共有101个数字
print(np.arange(12))

X = np.arange(12).reshape((4, 3)) # 把0~11十二个数的向量reshape成4行3列的矩阵

print(X)

print(X.reshape(3, 4)) # 把矩阵X调整成3行4列
import matplotlib.pyplot as plt # 导入matplotlib库
print(np.exp([0,1,2])) # 求自然底数e的幂
x = np.array([1,2,3])

print(x.shape)

y = np.array([[1,2,3]]) # 注意与y = np.array([1, 2, 3])的区别，X是维度为3的向量，Y是维度为1行3列的矩阵

print(y.shape)

print(x)

print(y)

print(x + y) # 注意x+y的结果是1行3列的矩阵
x = np.linspace(-10, 10, 201) # -10到10，等距离划分200个点

y = 1 / (1 + np.exp(-x))

plt.title("happy coding") # 图片标题

plt.xlabel("time") # 图片的横轴标题

plt.ylabel("speed") # 图片的纵轴标题

plt.plot(x, y, '-') # 用连续的-线画图

plt.show() # 展示图片

plt.plot(x, y, 'r-') # 'r-'代表红色的线，你还可以试试'bo', 'g--'

plt.show() # 展示图片
plt.plot([-2,-1,0,1], [0,4,1,3], 'x') # 'x'表示散点图

plt.show()

plt.plot([-2,-1,0,1], [0,4,1,3], 'x--') # 'x'表示散点图，'--'表示用虚线把散点图连成折线图

plt.show()
# 产生10000个 从均值为100，方差为20的高斯分布 随机生成的点

mu = 100

sigma = 20



# “均值为mu方差，为sigma高斯分布” = mu + sigma * “均值为0，方差为1的高斯分布”

# 这是高斯分布的数学性质

x = mu + sigma * np.random.randn(10000)

print(x)



# 柱状图统计出现次数

plt.hist(x, 50) # 将数组x分为50个统计区间，统计每个区间中数字出现的次数，将出现次数画成柱状图

plt.show() # 可以看到图像还是比较像高斯（正态）分布的
import os

os.listdir(".") # 当前文件夹下所有文件

os.listdir("../input/youthaiimageclassification")
mnist_dir = "../input/youthaiimageclassification/"

f = open(mnist_dir + "mnist_train.csv")

print(f.readline())

print(f.readline())
for i in range(10):

    data_line = f.readline() # 读取一行

    

    # int(x) 将list中的每一个元素转换为整数

    # data_line.split(',') 用逗号分开，得到一个list

    # reshape 将图片转为28*28的矩阵

    image = np.array([int(x) for x in data_line.split(',')][1:]).reshape(28, 28) 

    

    # 使用imshow将其画出，注意这里cmap; 画的时候默认是彩色图，cmap="gray"转换为灰度图，cmap="gray_r"代表反灰度图（即0代表白色，255代表黑色）

    plt.imshow(image, cmap="gray_r") 

    

    plt.show() #只展示一张，需要再用plt.show()出现下一张
ans = []

for x in data_line.split(','):

    ans.append(int(x))

print(ans)



# 与下面一行等效

print([int(x) for x in data_line.split(',')])
import pandas as pd

mnist_train = pd.read_csv(mnist_dir + "/mnist_train.csv")

mnist_test = pd.read_csv(mnist_dir + "/mnist_test.csv")

x_train = np.array(mnist_train.iloc[:, 1:]).reshape(-1, 28, 28) # 选取所有行，从第二列到最后一列（28*28=784个像素的值）

y_train = np.array(mnist_train.iloc[:, 0])

x_test = np.array(mnist_test.iloc[:, 1:]).reshape(-1, 28, 28)

y_test = np.array(mnist_test.iloc[:, 0])

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
class Cat:

    def __init__(self, age, name):

        self.age = age

        self.name = name

        

    def shout(self):

        print(self.name)
cat1 = Cat(5, "haha")

cat1.shout()

cat2 = Cat(6, "hehe")

cat2.shout()
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

mnist_train = pd.read_csv(mnist_dir + "/mnist_train.csv")

mnist_test = pd.read_csv(mnist_dir + "/mnist_test.csv")

x_train = np.array(mnist_train.iloc[:, 1:]).reshape(-1, 28, 28) # 选取所有行，从第二列到最后一列（28*28=784个像素的值）

y_train = np.array(mnist_train.iloc[:, 0])

x_test = np.array(mnist_test.iloc[:, 1:]).reshape(-1, 28, 28)

y_test = np.array(mnist_test.iloc[:, 0])

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
x_train = x_train.reshape(x_train.shape[0], -1)

x_test = x_test.reshape(x_test.shape[0], -1)



# 训练过程

knc = KNeighborsClassifier(n_neighbors=5)



n_train = 2000

knc.fit(x_train[:n_train], y_train[:n_train]) # fit是训练的函数 predict是测试的函数



# 测试过程

n_test = 100

np.sum(knc.predict(x_test[0:100]) == y_test[0:100].flatten()) # predict是测试的函数

# 输出100个数据中预测正确的个数
# 画出分类错误的图像

print(np.where(knc.predict(x_test[0:100]) != y_test[0:100])[0])

for i in np.where(knc.predict(x_test[0:100]) != y_test[0:100])[0]:

    plt.imshow(x_test[i].reshape(28, 28), cmap="gray_r")

    plt.show()
import pickle

with open("../input/youthaiimageclassification/cifar10.pkl", "rb") as f:

    (x_train, y_train), (x_test, y_test) = pickle.load(f)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
x_train = x_train.reshape(x_train.shape[0], -1)

x_test = x_test.reshape(x_test.shape[0], -1)



# 训练过程

knc = KNeighborsClassifier(n_neighbors=5)



n_train = 2000

knc.fit(x_train[:n_train], y_train[:n_train]) # fit是训练的函数 predict是测试的函数



# 测试过程

n_test = 100

print(knc.predict(x_test[0:100]) == y_test[0:100].flatten())

print(np.sum(knc.predict(x_test[0:100]) == y_test[0:100].flatten())) # predict是测试的函数

# 输出100个数据中预测正确的个数
# 画出分类错误的图像

for i in np.where(knc.predict(x_test[0:100]) != y_test[0:100].flatten())[0]:

    plt.imshow(x_test[i].reshape(32, 32, 3))

    plt.show()

    if i>30:

        break