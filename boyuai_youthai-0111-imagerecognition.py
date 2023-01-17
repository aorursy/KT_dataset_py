import numpy as np # 导入numpy库

import matplotlib.pyplot as plt # 导入matplotlib库
x = np.array([1, 2, 3])

y = 2 * x + 1
plt.plot(x, y, 'x-')
w = 0

b = 0

def predict(x):

    return w * x + b
y_predict = predict(x)



def MSE(y_predict, y):

    return 0.5 * np.mean((y_predict - y) ** 2)# 提示：np.mean的作用就是求平均，这里是把所有的平方差加起来求平均
MSE(predict(x), y)  # 输出一下当前w和b下误差是多少，尝试不同的 w和 b 
w = 0

b = 0

learning_rate = .01               # 学习率

loss_list = []                    # 保存训练过程中每次更新后的loss

for i in range(50):

    d_w = (y - predict(x)) @ -x   # w的梯度

    d_b = np.sum(predict(x) - y)  # b的梯度

    w -= learning_rate * d_w      # 更新w(为了降低loss则减去梯度，增加loss则加上梯度)

    b -= learning_rate * d_b      # 更新b

    loss = MSE(predict(x), y)     # 求出新的loss

    loss_list.append(loss)        # 记录loss，以便后面可视化

    print(w, b, loss)             # 输出更新后的w,b以及loss
plt.plot(loss_list)                
def f(x):                    # 目标(损失)函数

    return x ** 2



def d(x):                    # 梯度 d(x²) = 2x

    return 2 * x



x = np.linspace(-5, 5, 100)     # 画出损失函数，在[-5，5]上取100个点, 用numpy函数

y = f(x)

plt.plot(x, y)

plt.title("loss")
from IPython.display import display, clear_output

x_start = -3              # 初始化参数

learning_rate = 0.1       # 学习率 1.1发散原理讲解

step = 10                 # 迭代步数

for i in range(step):

    x_start = x_start - learning_rate * d(x_start)         # 用梯度更新参数

    plt.title("x: %.4f, y: %.4f" % (x_start, f(x_start)))  # 标题 x, y保留四位小数

    plt.plot(x, y)                                         # 画出损失函数

    plt.plot(x_start, f(x_start), 'ro')                    # 画出当前的参数值以及对应的损失函数 

    plt.show()

    clear_output(wait=True)                                # 等到下一张图来了再删除上一张

    plt.pause(0.5)                                         # 每张图停留0.5s
def f(x):                    # 目标(损失)函数               

    return x ** 4 + x ** 3 - 20 * x ** 2 + x + 1



def d(x):                    # 梯度 d(x) 

    return 4 * x ** 3 + 3 * x ** 2 - 40 * x + 1



x = np.linspace(-5, 5, 100)  # 画出损失函数                

y = f(x)

plt.plot(x, y)

plt.title("loss")
from IPython.display import display, clear_output   # 尝试不同学习率的结果            

x_start = -2             # 初始化参数 尝试 1, -2 

learning_rate = 0.01      # 学习率 0.01

step = 10                 # 迭代步数

for i in range(step):

    x_start = x_start - learning_rate * d(x_start)         # 用梯度更新参数

    plt.title("x: %.4f, y: %.4f" % (x_start, f(x_start)))  # 标题 x, y保留四位小数

    plt.plot(x, y)                                         # 画出损失函数

    plt.plot(x_start, f(x_start), 'ro')                    # 画出当前的参数值以及对应的损失函数 

    plt.show()

    clear_output(wait=True)                                # 等到下一张图来了再删除上一张

    plt.pause(0.5)  
from keras.models import Sequential                               # 序列模型，线性逐层叠加

from keras.layers import Dense, Activation, Flatten, Dropout      # 导入全连接层、激活函数层、二维转一维、Dropout等神经网络常用层

from keras.optimizers import SGD                                  # 导入随机梯度下降优化器



import numpy as np # 导入numpy库

import matplotlib.pyplot as plt # 导入matplotlib库
x = np.array([

    [0, 0],

    [0, 1],

    [1, 0],

    [1, 1]

])

y = np.array([1, 0, 0, 1])



model = Sequential()                  # 创建一个序列模型的对象          

model.add(Dense(2))                   # 添加一个全连接层，2->2

model.add(Activation('sigmoid'))      # 添加一个sigmoid的激活函数

model.add(Dense(1))                   # 继续添加一个全连接层, 2->1

model.add(Activation('sigmoid'))      # 再添加一个sigmoid的激活函数
model.compile(optimizer = SGD(lr = 1),      # 选用sgd的优化器，学习率设为1

              loss = 'binary_crossentropy', # 损失(目标)函数采用二分类交叉熵损失函数

              metrics = ['accuracy'])       # 用精度作为性能评价指标
history = model.fit(x, y, epochs=1000)    # 模型训练，history记录了训练过程中的一些中间信息
plt.plot(history.history["loss"])       # 画出训练过程中每一步精度的变化

plt.plot(history.history["accuracy"])        # 画出训练过程中每一步精度的变化
for layer in model.layers:        # 遍历模型的所有层

    print(layer.get_weights())    # 输出每层的权重(即需要训练的参数)
import pandas as pd                                                  # 导入pandas库用于读取数据

import numpy as np  # 导入numpy库

# mnist_dir = "youthai/data/mnist-in-csv/"

mnist_dir = "../input/youthaiimageclassification/"

mnist_train = pd.read_csv(mnist_dir + "mnist_train.csv")   #数据读取，当你的input只有mnist数据集时，目录应改为‘../input/mnist_train.csv’

mnist_test = pd.read_csv(mnist_dir + "mnist_test.csv")     #具体路径最好通过 os.listdir('../input/') 看下

# mnist_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")   # 数据读取，当你的input只有mnist数据集时，目录应改为‘../input/mnist_train.csv’

# mnist_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")     # 具体路径最好通过 os.listdir('../input/') 看下

x_train = np.array(mnist_train.iloc[:, 1:]).reshape(-1, 28, 28)      # 去除训练数据第一列的标签数据，并将数据reshape成 N×h×w

y_train = np.array(mnist_train.iloc[:, 0])                           # 提取训练数据标签

x_test = np.array(mnist_test.iloc[:, 1:]).reshape(-1, 28, 28)        # 去除测试数据第一列的标签数据，并将数据reshape成 N×h×w

y_test = np.array(mnist_test.iloc[:, 0])                             # 提取测试数据标签



num_classes = 10    # 识别手写数字这个问题中一共有0~9个数字，共10类
from keras.models import Sequential                               # 序列模型，线性逐层叠加

from keras.layers import Dense, Activation, Flatten, Dropout      # 导入全连接层、激活函数层、二维转一维、Dropout等神经网络常用层

from keras.optimizers import SGD                                  # 导入随机梯度下降优化器

import matplotlib.pyplot as plt # 导入matplotlib库

from keras.utils import to_categorical



x_train = x_train / 255        # 数据归一化

x_test = x_test / 255          

y_train = to_categorical(y_train, num_classes)  # 将标签变成独热编码，方便后面的交叉熵计算

y_test = to_categorical(y_test, num_classes)
model = Sequential()                           # 创建一个序列模型的对象

model.add(Flatten(input_shape = (28, 28)))     # 对于每个数据(这里指图片)全连接输入必输为一个向量，因此使用Flatten层起到了reshape的作用

model.add(Dense(20, activation = 'relu'))      # 加上全连接层 784->20

model.add(Dense(20, activation = 'relu'))      # 加上全连接层 20->20

model.add(Dense(num_classes, activation = 'softmax')) # 添加全连接层，然后加上softmax的激活函数
model = Sequential()                           # 创建一个序列模型的对象

model.add(Flatten(input_shape = (28, 28)))     # 对于每个数据(这里指图片)全连接输入必输为一个向量，因此使用Flatten层起到了reshape的作用

model.add(Dense(512, activation = 'relu'))     # 加上全连接层 784->512

model.add(Dropout(0.2))                        # dropout层在每一个batchsize训练中随机使网络中一些节点失效(0.2的概率)，可以起到解耦合，防止过拟合等一系列作用

model.add(Dense(512, activation = 'relu'))     # 加上全连接层 512->512

model.add(Dropout(0.2))                        # 添加dropout层

model.add(Dense(num_classes, activation = 'softmax')) # 添加全连接层，然后加上softmax的激活函数
model.summary()  # 输出模型各层详细信息，可以看到各层参数状况
model.compile(loss='categorical_crossentropy',   # 损失函数使用多类交叉熵损失函数

              optimizer="rmsprop",               # 优化器采用rmsprop

              metrics=['accuracy'])              # 用精度作为性能评估指标
batch_size = 32        # 每次输入32张图片,前向传播求出损失函数平均值，然后反向传播一次更新梯度

epochs = 5             # 保证所有训练数据被输入网络五次

history = model.fit(x_train, y_train, # 训练数据

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,   # 越大，训练过程中显示的信息越详细  查bing

                    validation_data=(x_test, y_test))  # 验证集

score = model.evaluate(x_test, y_test, verbose=0)      # 模型评估，返回模型的loss和metric

print('Test loss:', score[0])     # 测试集上模型损失

print('Test accuracy:', score[1]) # 测试集上模型精度
model.predict(x_test[0:1]) # 模型预测，输出预测的标签信息
import pickle

with open("../input/youthaiimageclassification/cifar10.pkl", "rb") as f:

    (x_train, y_train), (x_test, y_test) = pickle.load(f)
from keras.models import Sequential                               # 序列模型，线性逐层叠加

from keras.layers import Dense, Activation, Flatten, Dropout      # 导入全连接层、激活函数层、二维转一维、Dropout等神经网络常用层

from keras.optimizers import SGD                                  # 导入随机梯度下降优化器

import matplotlib.pyplot as plt # 导入matplotlib库

from keras.utils import to_categorical



# (x_train, y_train), (x_test, y_test) = cifar10.load_data()   # 用keras提供的api读取数据

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

x_train = x_train / 255  # 数据归一化

x_test = x_test / 255

num_classes = 10         # 数据一共有10类

y_train = to_categorical(y_train, num_classes) # 将训练数据的标签独热编码

y_test = to_categorical(y_test, num_classes)   # 将测试数据的标签独热编码
model = Sequential()                          # 创建序列模型的对象

model.add(Flatten(input_shape=(32, 32, 3)))   # 讲解shape，用Flatten层将数据reshape成batchsize×（32*32*3）

model.add(Dense(512, activation='relu'))      # 添加全连接层，使用relu作为激活函数3072->512

model.add(Dense(512, activation='relu'))      # 添加全连接层，使用relu作为激活函数512->512

model.add(Dense(num_classes, activation='softmax'))# 添加全连接层，激活函数为softmax 512->10



model.compile(loss='categorical_crossentropy',  # 多类交叉熵损失函数

              optimizer="rmsprop",              # 优化器使用rmsprop

              metrics=['accuracy'])             # 评估指标：精度
batch_size = 32               # 每次输入32张图片,前向传播求出损失函数平均值，然后反向传播一次更新梯度

epochs = 5                    # 保证所有训练数据被输入网络五次

history = model.fit(x_train, y_train,                   # 训练数据

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,                          # 越大，训练过程中显示的信息越详细             

                    validation_data=(x_test, y_test))   # 验证集

score = model.evaluate(x_test, y_test, verbose=0)       # 模型评估，返回模型的loss和metric

print('Test loss:', score[0])                           # 测试集上模型损失

print('Test accuracy:', score[1])                       # 测试集上模型精度
model = Sequential()                          # 创建序列模型的对象

model.add(Flatten(input_shape=(32, 32, 3)))   # 讲解shape，用Flatten层将数据reshape成batchsize×（32*32*3）

model.add(Dense(512, activation='relu'))      # 添加全连接层，使用relu作为激活函数3072->512

model.add(Dropout(0.2))                       # 添加dropout层，dropout层在每一个batchsize训练中随机使网络中一些节点失效(0.2的概率)，可以起到解耦合，防止过拟合等一系列作用

model.add(Dense(512, activation='relu'))      # 添加全连接层，使用relu作为激活函数512->512

model.add(Dropout(0.2))                       # 添加dropout层

model.add(Dense(num_classes, activation='softmax'))# 添加全连接层，激活函数为softmax 512->10



model.compile(loss='categorical_crossentropy',  # 多类交叉熵损失函数

              optimizer="rmsprop",              # 优化器使用rmsprop

              metrics=['accuracy'])             # 评估指标：精度

batch_size = 32               # 每次输入32张图片,前向传播求出损失函数平均值，然后反向传播一次更新梯度

epochs = 5                    # 保证所有训练数据被输入网络五次

history = model.fit(x_train, y_train,                   # 训练数据

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,                          # 越大，训练过程中显示的信息越详细             

                    validation_data=(x_test, y_test))   # 验证集

score = model.evaluate(x_test, y_test, verbose=0)       # 模型评估，返回模型的loss和metric

print('Test loss:', score[0])                           # 测试集上模型损失

print('Test accuracy:', score[1])                       # 测试集上模型精度
x_train.shape
import pickle

with open("../input/youthaiimageclassification/cifar10.pkl", "rb") as f:

    (x_train, y_train), (x_test, y_test) = pickle.load(f)



    

from keras.models import Sequential                               # 序列模型，线性逐层叠加

from keras.layers import Dense, Activation, Flatten, Dropout      # 导入全连接层、激活函数层、二维转一维、Dropout等神经网络常用层

from keras.optimizers import SGD                                  # 导入随机梯度下降优化器

import matplotlib.pyplot as plt # 导入matplotlib库

from keras.utils import to_categorical



# 数据预处理

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

x_train = x_train / 255  # 数据归一化

x_test = x_test / 255

num_classes = 10         # 数据一共有10类

y_train = to_categorical(y_train, num_classes) # 将训练数据的标签独热编码

y_test = to_categorical(y_test, num_classes)   # 将测试数据的标签独热编码

from keras.layers import Conv2D, MaxPooling2D  # 从keras导入卷积层和最大池化层



model = Sequential()

model.add(Conv2D(16, (5, 5), padding='same',    # 添加卷积层；16：卷积核的个数;（5，5）:卷积核大小；padding=’same‘：图片卷积后大小不变

                 input_shape=x_train.shape[1:]))# 第一个卷基层需要告诉它输入图片大小，以方便网络推导后面所需参数

model.add(Activation('relu'))                   # 使用relu作为激活函数

model.add(Conv2D(32, (5, 5)))                   # 添加卷积层

model.add(Activation('sigmoid'))                # 使用sigmoid作为激活函数

model.add(MaxPooling2D(pool_size=(2, 2)))       # 最大池化层，在2*2的区域中选取最大的数

model.add(Dropout(0.25))                        # 添加dropout层，dropout层在每一个batchsize训练中随机使网络中一些节点失效(0.25的概率)



model.add(Conv2D(64, (5, 5), padding='same'))   # 添加卷积层；64：卷积核的个数;（5，5）:卷积核大小；padding=’same‘：图片卷积后大小不变

model.add(Activation('relu'))                   # 使用relu作为激活函数

model.add(MaxPooling2D(pool_size=(2, 2)))       # 最大池化层，在2*2的区域中选取最大的数

model.add(Dropout(0.25))                        # 添加dropout层，dropout层在每一个batchsize训练中随机使网络中一些节点失效(0.25的概率)



model.add(Flatten())

model.add(Dense(100))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))



# 模型编译

model.compile(loss='categorical_crossentropy',  # 损失函数使用多类交叉熵损失函数

              optimizer="adam",                 # 优化器采用adam

              metrics=['accuracy'])             # 用精度作为性能评价指标
from keras.layers import Conv2D, MaxPooling2D  # 从keras导入卷积层和最大池化层



model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',    # 添加卷积层；32：卷积核的个数;（3，3）:卷积核大小；padding='same'：图片卷积后大小不变

                 input_shape=x_train.shape[1:]))# 第一个卷基层需要告诉它输入图片大小，以方便网络推导后面所需参数

model.add(Activation('relu'))                   # 使用relu作为激活函数

model.add(Conv2D(32, (3, 3)))                   # 添加卷积层

model.add(Activation('relu'))                   # 使用relu作为激活函数

model.add(MaxPooling2D(pool_size=(2, 2)))       # 最大池化层，在2*2的区域中选取最大的数

model.add(Dropout(0.25))                        # 添加dropout层，dropout层在每一个batchsize训练中随机使网络中一些节点失效(0.25的概率)



model.add(Conv2D(64, (3, 3), padding='same'))   # 添加卷积层；64：卷积核的个数;（3，3）:卷积核大小；padding='same'：图片卷积后大小不变

model.add(Activation('relu'))                   # 使用relu作为激活函数

model.add(Conv2D(64, (3, 3)))                   # 添加卷积层；64：卷积核的个数;（3，3）

model.add(Activation('relu'))                   # 使用relu作为激活函数

model.add(MaxPooling2D(pool_size=(2, 2)))       # 最大池化层，在2*2的区域中选取最大的数

model.add(Dropout(0.25))                        # 添加dropout层，dropout层在每一个batchsize训练中随机使网络中一些节点失效(0.25的概率)



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))



# 模型编译

model.compile(loss='categorical_crossentropy',  # 损失函数使用多类交叉熵损失函数

              optimizer="adam",                 # 优化器采用adam

              metrics=['accuracy'])             # 用精度作为性能评价指标
batch_size = 32               # 每次输入32张图片,前向传播求出损失函数平均值，然后反向传播一次更新梯度

epochs = 5                    # 保证所有训练数据被输入网络五次

history = model.fit(x_train, y_train,                   # 训练数据

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,                          # 越大，训练过程中显示的信息越详细             

                    validation_data=(x_test, y_test))   # 验证集

score = model.evaluate(x_test, y_test, verbose=0)       # 模型评估，返回模型的loss和metric

print('Test loss:', score[0])                           # 测试集上模型损失

print('Test accuracy:', score[1])                       # 测试集上模型精度
from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt
import pickle

with open("../input/youthaiimageclassification/cifar10.pkl", "rb") as f:

    (x_train, y_train), (x_test, y_test) = pickle.load(f)



    

from keras.models import Sequential                               # 序列模型，线性逐层叠加

from keras.layers import Dense, Activation, Flatten, Dropout      # 导入全连接层、激活函数层、二维转一维、Dropout等神经网络常用层

from keras.optimizers import SGD                                  # 导入随机梯度下降优化器

import matplotlib.pyplot as plt # 导入matplotlib库

from keras.utils import to_categorical



# 数据预处理

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

x_train = x_train / 255  # 数据归一化

x_test = x_test / 255

num_classes = 10         # 数据一共有10类

y_train = to_categorical(y_train, num_classes) # 将训练数据的标签独热编码

y_test = to_categorical(y_test, num_classes)   # 将测试数据的标签独热编码

datagen = ImageDataGenerator(

    rotation_range=30,

    horizontal_flip=True,

    vertical_flip=True,

    width_shift_range=5,

    height_shift_range=5

)
origin_image = x_train[1] # 选取原图



# 将原图画出来

plt.imshow(origin_image) 

plt.show()



# 对图像作五次随机变换并画出来

fig, ax = plt.subplots(1, 5, figsize=(15, 3))

ax = ax.flatten()

for i in range(5):

    ax[i].imshow(datagen.random_transform(origin_image)) # 使用datagen对图像作随机变换

plt.show()
model.fit(x_train, y_train, batch_size=32, epochs=10)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs=10)
plt.imshow(origin_image)

plt.show()
transformed_image = datagen.apply_transform(origin_image, {

    "theta": 30

})

plt.imshow(transformed_image)

plt.show()
transformed_image = datagen.apply_transform(origin_image, {

    "tx": 5,

    "ty": 5

})

plt.imshow(transformed_image)

plt.show()
transformed_image = datagen.apply_transform(origin_image, {

    "flip_vertical": True

})

plt.imshow(transformed_image)

plt.show()
transformed_image = datagen.apply_transform(origin_image, {

    "flip_horizontal": True

})

plt.imshow(transformed_image)

plt.show()