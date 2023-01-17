import os

print(os.listdir("../input"))
f = open("../input/mnist_train.csv")

print(f.readline())

print(f.readline())
import matplotlib.pyplot as plt

import numpy as np



data = f.readline() # 读取一行

data_split = data.split(',') # 用逗号分开，得到一个list

image = [int(x) for x in data_split[1:]] # 将list中的每一个元素转换为整数

image = np.array(image) # 将list转换成numpy数组

image = image.reshape(28, 28) # 将图片转为28*28的矩阵

plt.imshow(image, cmap="gray_r") # 使用imshow将其画出，注意这里cmap="gray_r"代表反灰度图（即0代表白色，255代表黑色）

plt.show()
for i in range(5):

    for j in range(4):

        image = np.array([int(x) for x in f.readline().split(',')[1:]]).reshape(28, 28) # 我们将之前的操作浓缩成了一行代码

        plt.subplot(5, 4, i*4+j+1) # 5行4列，第三个参数代表当前是第几幅图（先按行再按列，比如第二行第一列就是第5幅图）

        plt.imshow(image, cmap="gray_r")

        plt.xticks([]) # 去掉x坐标轴的刻度

        plt.yticks([]) # 去掉y坐标轴的刻度
x = np.array([1, 2, 1, 0, 3])

y = np.array([-1, -1, 2, 1, 1])



print(x - y) # 对位相减

print(x + y) # 对位相加

print(x * y) # 对位相乘

print(x @ y) # 点积

print(x ** 2) # 每一位平方
A = np.array([

    [1, 0, 2],

    [2, 1, 0],

    [2, 0, 1]

])

B = np.array([

    [2, 3, 0],

    [1, 0, 1],

    [0, 3, 2]

])

C = np.array([

    [1, 2],

    [0, 1],

    [2, 2]

])



print((A + B) @ C) # 矩阵相加然后相乘
print(np.zeros(10))

print(np.ones(10))

print(np.ones((5, 5))) 

print(np.random.randn(5, 5)) # 用均值为0，方差为1的高斯分布随机生成5*5的矩阵，
import pandas as pd

mnist_train = pd.read_csv("../input/mnist_train.csv")

mnist_test = pd.read_csv("../input/mnist_test.csv")

x_train = np.array(mnist_train.iloc[:, 1:]).reshape(-1, 28, 28) # 选取所有行，从第二列到最后一列（28*28=784个像素的值）

y_train = np.array(mnist_train.iloc[:, 0])

x_test = np.array(mnist_test.iloc[:, 1:]).reshape(-1, 28, 28)

y_test = np.array(mnist_test.iloc[:, 0])

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
for position in np.where(y_train == 5)[0][0:10]:

    plt.imshow(x_train[position], cmap="gray_r")

    plt.show()
x = np.linspace(-5, 5, 100) # -5到5，等距离划分100个点 

y = x ** 2 # y=x^2
plt.title("hello boyu")

plt.xlabel("x")

plt.ylabel("y")

plt.plot(x, y, 'r-') # 'r-'代表红色的线，你还可以试试'bo', 'g--'

plt.show()
# 产生10000个 从均值为100，方差为20的高斯分布 随机生成的点

mu = 100

sigma = 20

x = mu + sigma * np.random.randn(10000)



# 柱状图统计出现次数

plt.hist(x, 50) # 分为50个统计区间
class Cat:

    def __init__(self, name):

        self.name = name

        

    def meow(self):

        print("I'm a cat. My name is %s" % self.name)

        

class Dog:

    def __init__(self, name, age):

        self.name = name

        self.age = age

        

    def __str__(self):

        return self.name

        

    def bark(self):

        print("My name is %s! My age is %d" % (self.name, self.age))

        

    

cat1 = Cat("kittie")

cat2 = Cat("Schediinger")

cat1.meow()



dog1 = Dog("lala", 16)

dog1.bark()

print(dog1) # 调用 dog1.__str__()
from sklearn.neighbors import KNeighborsClassifier



x_train = x_train.reshape(x_train.shape[0], -1)

x_test = x_test.reshape(x_test.shape[0], -1)

print(x_train.shape)

print(x_test.shape)
k = 5

knc = KNeighborsClassifier(n_neighbors=5)
knc.fit(x_train, y_train)
# 用训练好的K近邻算法分类x_test中的100张图片

y_predict = knc.predict(x_test[0:100])

print("准确率为", np.sum(y_predict == y_test[0:100]) / 100)
# 画出分类错误的图像

for i in np.where(y_predict != y_test[0:100])[0]:

    print("预测分类是%d" % y_predict[i])

    print("正确分类是%d" % y_test[i])

    plt.imshow(x_test[i].reshape(28, 28), cmap="gray_r")

    plt.show()