import pickle

with open("../input/youthaiimageclassification/cifar10.pkl", "rb") as f:

    (x_train_read, y_train_read), (x_test_read, y_test_read) = pickle.load(f)
from keras.utils import to_categorical



print(x_train_read.shape)

print(y_train_read.shape)

print(x_test_read.shape)

print(y_test_read.shape)

x_train = x_train_read / 255  # 数据归一化

x_test = x_test_read / 255

num_classes = 10         # 数据一共有10类

y_train = to_categorical(y_train_read, num_classes) # 将训练数据的标签独热编码

y_test = to_categorical(y_test_read, num_classes)   # 将测试数据的标签独热编码



print(y_train.shape)

print(y_test.shape)
from keras.models import Sequential           # 序列模型，线性逐层叠加

from keras.layers import Dense, Flatten       # 导入全连接层、激活函数层、二维转一维、Dropout等神经网络常用层



model = Sequential()                          # 创建序列模型的对象

model.add(Flatten(input_shape=(32, 32, 3)))   # 讲解shape，用Flatten层将数据reshape成batchsize×（32*32*3）

model.add(Dense(512, activation='relu'))      # 添加全连接层，使用relu作为激活函数3072->512

model.add(Dense(512, activation='relu'))      # 添加全连接层，使用relu作为激活函数512->512

model.add(Dense(num_classes, activation='softmax'))# 添加全连接层，激活函数为softmax 512->10



model.compile(loss='categorical_crossentropy',  # 多类交叉熵损失函数

              optimizer="rmsprop",              # 优化器使用rmsprop

              metrics=['accuracy'])             # 评估指标：精度

model.summary()
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
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation  # 从keras导入卷积层、最大池化层、Dropout层和激活层



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

model.summary()
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