import numpy as np

import matplotlib.pyplot as plt
x = np.array([1, 2, 3])

y = 2 * x + 1
w = 0

b = 0

def predict(x):

    return w * x + b
def MSE(y_predict, y):

    return 0.5 * np.mean((y_predict - y) ** 2) # np.mean的作用就是把所有的平方差加起来求平均
MSE(predict(x), y)  #输出一下当前w和b下误差是多少
w = 0

b = 0

learning_rate = .01

loss_list = []                    #保存训练过程中每次更新后的loss

for i in range(50):

    d_w = (y - predict(x)) @ -x   #w的梯度

    d_b = np.sum(predict(x) - y)  #b的梯度

    w -= learning_rate * d_w      #更新w(为了降低loss则减去梯度，增加loss则加上梯度)

    b -= learning_rate * d_b      #更新b

    loss = MSE(predict(x), y)     #求出新的loss

    loss_list.append(loss)        #记录loss，以便后面可视化

    print(w, b, loss)             #输出更新后的w,b以及loss
plt.plot(loss_list)                #可视化loss每一步变化
def f(x):                    #目标(损失)函数

    return x ** 2



def d(x):                    #梯度

    return 2 * x



x = np.linspace(-5, 5, 100)  #画出损失函数

y = f(x)

plt.plot(x, y)
from IPython.display import display, clear_output

x_start = -3              #初始化参数

learning_rate = 0.1       #学习率

step = 10                 #迭代步数

for i in range(step):

    x_start = x_start - learning_rate * d(x_start)    #用梯度更新参数

    plt.title("x: %.4f, y: %.4f" % (x_start, f(x_start)))  #标题

    plt.plot(x, y)                                         #画出损失函数

    plt.plot(x_start, f(x_start), 'ro')                    #画出当前的参数值以及对应的损失函数 

    plt.show()

    clear_output(wait=True)                                #等到下一张图来了再删除上一张

    plt.pause(10)                                          #每张图停留0.5s
from keras.models import Sequential                               #序列模型，线性逐层叠加

from keras.layers import Dense, Activation, Flatten, Dropout

from keras.optimizers import SGD
x = np.array([

    [0, 0],

    [0, 1],

    [1, 0],

    [1, 1]

])

y = np.array([0, 0, 0, 1])



model = Sequential()                  #创建一个序列模型的对象          

model.add(Dense(2))                   #添加一个全连接层，2->2

model.add(Activation('sigmoid'))      #添加一个sigmoid的激活函数

model.add(Dense(1))                   #继续添加一个全连接层, 2->1

model.add(Activation('sigmoid'))      #再添加一个sigmoid的激活函数



#编译模型

model.compile(optimizer=SGD(lr=1),    #选用sgd的优化器，学习率设为1

              loss='binary_crossentropy', #损失(目标)函数采用二分类交叉熵损失函数

              metrics=['accuracy'])       #用精度作为性能评价指标
history = model.fit(x, y, epochs=500)    #模型训练，history记录了训练过程中的一些中间信息
plt.plot(history.history["loss"])       #画出训练过程中每一步loss的变化

plt.plot(history.history["acc"])        #画出训练过程中每一步精度的变化
for layer in model.layers:        #遍历模型的所有层

    print(layer.get_weights())    #输出每层的权重(即需要训练的参数)
import pandas as pd

import numpy as np

mnist_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")   #数据读取，当你的input只有mnist数据集时，目录应改为‘../input/mnist_train.csv’

mnist_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")     #具体路径最好通过 os.listdir('../input/') 看下

x_train = np.array(mnist_train.iloc[:, 1:]).reshape(-1, 28, 28)      #去除训练数据第一列的标签数据，并将数据reshape成 N×h×w

y_train = np.array(mnist_train.iloc[:, 0])                           #提取训练数据标签

x_test = np.array(mnist_test.iloc[:, 1:]).reshape(-1, 28, 28)        #去除测试数据第一列的标签数据，并将数据reshape成 N×h×w

y_test = np.array(mnist_test.iloc[:, 0])                             #提取测试数据标签



num_classes = 10    #这个问题中一共有10类
from keras.utils import to_categorical

x_train = x_train / 255        #数据归一化

x_test = x_test / 255          

y_train = to_categorical(y_train, num_classes)  #将标签变成独热编码，方便后面的交叉熵计算

y_test = to_categorical(y_test, num_classes)
model = Sequential()                         #创建一个序列模型的对象

model.add(Flatten(input_shape=(28, 28)))     #对于每个数据(这里指图片)全连接输入必输为一个向量，因此使用Flatten层起到了reshape的作用

model.add(Dense(512, activation='relu'))     #加上全连接层 784->512

model.add(Dropout(0.2))                      #dropout层在每一个batchsize训练中随机使网络中一些节点失效(0.2的概率)，可以起到解耦合，防止过拟合等一系列作用

model.add(Dense(512, activation='relu'))     #加上全连接层 512->512

model.add(Dropout(0.2))                      #添加dropout层

model.add(Dense(num_classes, activation='softmax')) #添加全连接层，然后加上softmax的激活函数
model.summary()  #输出模型各层详细信息，可以看到各层参数状况
model.compile(loss='categorical_crossentropy',   #损失函数使用多类交叉熵损失函数

              optimizer="rmsprop",               #优化器采用rmsprop

              metrics=['accuracy'])              #用精度作为性能评估指标
batch_size = 32        #每次输入32张图片,前向传播求出损失函数平均值，然后反向传播一次更新梯度

epochs = 5             #保证所有训练数据被输入网络五次

history = model.fit(x_train, y_train, #训练数据

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,   #越大，训练过程中显示的信息越详细

                    validation_data=(x_test, y_test))  #验证集

score = model.evaluate(x_test, y_test, verbose=0)  #模型评估，返回模型的loss和metric

print('Test loss:', score[0])  #测试集上模型损失

print('Test accuracy:', score[1]) #测试集上模型精度
model.predict(x_test[0:1]) #模型预测，输出预测的标签信息
from os import listdir, makedirs

from os.path import join, exists, expanduser



cache_dir = expanduser(join('~', '.keras'))  #join：将两个字符串合成一个路径；expanduser：把path中包含的"~"和"~user"转换成用户目录，这里~为/tmp/

if not exists(cache_dir):   #检测路径是否存在

    makedirs(cache_dir)     #不存在的情况下创建路径

datasets_dir = join(cache_dir, 'datasets') 

if not exists(datasets_dir):

    makedirs(datasets_dir)           #最终创建路径为/tmp/.keras/datasets/

    

# if not exists(join(cache_dir,'datasets')):

#     makedirs(join(cache_dir,'datasets'))





# kaggle kernel可以使用linux shell命令，只需要在命令前加上感叹号

!cp ../input/cifar10-python/cifar-10-python.tar.gz ~/.keras/datasets/    #将/kaggle/input/cifar10-python/下的cifar-10-python.tar.gz文件拷贝到刚创建的/tmp/.keras/datasets目录下

!ln -s  ~/.keras/datasets/cifar-10-python.tar.gz ~/.keras/datasets/cifar-10-batches-py.tar.gz #为cifar-10-batches-py.tar.gz创建名字为cifar-10-python.tar.gz的符号链接

!tar xzvf ~/.keras/datasets/cifar-10-python.tar.gz -C ~/.keras/datasets/ #将cifar-10-python.tar.gz解压
!ls /tmp/.keras/datasets/cifar-10-batches-py/
from keras.datasets import cifar10

from keras.utils import to_categorical



(x_train, y_train), (x_test, y_test) = cifar10.load_data()   #用keras提供的api读取数据

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

x_train = x_train / 255  #数据归一化

x_test = x_test / 255

num_classes = 10         #数据一共有10类

y_train = to_categorical(y_train, num_classes) #将训练数据的标签独热编码

y_test = to_categorical(y_test, num_classes)   #将测试数据的标签独热编码
from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten, Dropout



model = Sequential()                          #创建序列模型的对象

model.add(Flatten(input_shape=(32, 32, 3)))   #用Flatten层将数据reshape成batchsize×（32*32*3）

model.add(Dense(512, activation='relu'))      #添加全连接层，使用relu作为激活函数3072->512

model.add(Dropout(0.2))                       #添加dropout层

model.add(Dense(512, activation='relu'))      #添加全连接层，使用relu作为激活函数512->512

model.add(Dropout(0.2))                       #添加dropout层

model.add(Dense(num_classes, activation='softmax'))#添加全连接层，激活函数为softmax 512->10



model.compile(loss='categorical_crossentropy',  #多类交叉熵损失函数

              optimizer="rmsprop",              #优化器使用rmsprop

              metrics=['accuracy'])             #评估指标：精度
batch_size = 32

epochs = 5

history = model.fit(x_train, y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,

                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
x_train.shape
from keras.layers import Conv2D, MaxPooling2D



model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',   #添加卷积层；32：卷积核的个数;（3，3）:卷积核大小；padding=’same‘：图片卷积后大小不变

                 input_shape=x_train.shape[1:]))#第一个卷基层需要告诉它输入图片大小，以方便网络推导后面所需参数

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))    #最大池化层，在2*2的区域中选取最大的数

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer="adam",

              metrics=['accuracy'])
batch_size = 32

epochs = 5

history = model.fit(x_train, y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,

                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])