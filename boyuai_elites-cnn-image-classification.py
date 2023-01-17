from keras.datasets import cifar10



(x_train, y_train), (x_test, y_test) = cifar10.load_data() #使用keras提供的api读取数据



num_classes = 10

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
import numpy as np

import matplotlib.pyplot as plt



fig, axes = plt.subplots(num_classes, 10, figsize=(15, 15)) # 新建一个包含100张子图的10行10列的画布

for i in range(num_classes): # 对于每一类物体

    indice = np.where(y_train == i)[0] #找到标签为i的图像下标

    for j in range(10): # 输出前10张图片

        axes[i][j].imshow(x_train[indice[j]])

        # 去除坐标刻度

        axes[i][j].set_xticks([]) 

        axes[i][j].set_yticks([])

plt.tight_layout()

plt.show()
from keras.utils import to_categorical



#归一化

x_train = x_train / 255

x_test = x_test / 255



#将训练数据的标签独热编码

y_train = to_categorical(y_train, num_classes)

y_test = to_categorical(y_test, num_classes)
print(y_train.shape)

print(y_train[0])
from keras.models import Sequential

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout



model = Sequential()

# Conv2D: 卷积层

# - 32: 卷积核的个数

# - (3, 3): 卷积核大小

# - padding=’same‘：补齐模式图片卷积后大小不变

# - input_shape=x_train.shape[1:]: 将输入大小告诉网络的第一层，方便推导后面所需参数

# - activation="relu": 使用 relu 激活函数

model.add(Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:], activation="relu")) 

model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))

model.add(MaxPooling2D(pool_size=2)) # 最大池化层，在2*2的区域中选取最大的数

model.add(Dropout(0.25)) # 丢弃层，随机将25%的神经元设为0



model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))

model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.25))



model.add(Flatten()) # 转为一维数据

model.add(Dense(512, activation="relu")) # 全连接层

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation="softmax")) # 最后一个全连接层使用 softmax 激活函数，输出分类概率



# 使用 model.compile 编译模型

# - loss='categorical_crossentropy': 使用交叉熵为损失函数

# - optimizer="adam": 使用 adam 优化器

# - metrics=['accuracy']: 使用准确率为指标

model.compile(loss='categorical_crossentropy',

              optimizer="adam",

              metrics=['accuracy'])
model.summary()
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