import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K

sns.set(style='white', context='notebook', palette='deep')

#加载训练数据
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

# free some space
del train 

g = sns.countplot(Y_train)

Y_train.value_counts()

X_train.isnull().any().describe()
test.isnull().any().describe()
# 图像数据做归一化（Normalize）处理
X_train = X_train / 255.0
test = test / 255.0

# 将图像数据转换成 28*28*1 格式 (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# 将label转换为one hot数据格式，为啥？1、稀疏矩阵运算快，2、最后网络的输出是[为1的概率,为2的概率，... ,为9的概率]，方便计算损失值
Y_train = to_categorical(Y_train, num_classes = 10)

# 切分训练集合测试集
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
g = plt.imshow(X_train[0][:,:,0])
# 下面是Lenet改进版的网络结构

model = Sequential()

# 第一个卷积块

# 5*5*32 第一个卷积层
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
# 5*5*32 第二个卷积层
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
# 2*2 池化
model.add(MaxPool2D(pool_size=(2,2)))
# 防过拟合，随机丢掉25%值
model.add(Dropout(0.25))


# 第二个卷积块

# 3*3*64 第一个卷积层
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
# 3*3*64 第二个卷积层
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
# 2*2 池化
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# 防过拟合，随机丢掉25%值
model.add(Dropout(0.25))

# 第一个全连接层，输出256个值
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
# 防过拟合，随机丢掉50%值
model.add(Dropout(0.5))

# 第二个全连接层，输出10个分类的概率
model.add(Dense(10, activation = "softmax"))
# 定义训练的优化器
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

epochs = 1 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86

# 执行训练
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, Y_val), verbose = 2)
# 获取测试数据
test_data = test[0]
test_data = test_data.reshape(1,(test.shape)[1],(test.shape)[2],1)
print(test_data.shape)

# 预测结果
predicted_labels = model.predict(test_data)
predicted_labels=np.round(predicted_labels,decimals=2)
print(predicted_labels)

# 打印输入的图形
g = plt.imshow(test[0][:,:,0],cmap='gray')

layer_0 = K.function([model.layers[0].input], [model.layers[0].output])
f0 = layer_0([test_data])[0]
print(f0.shape)

#第一层卷积块第一次卷积后的特征图展示，输出是（1,28,28,32）
for _ in range(32):
    show_img = f0[:, :, :, _]
    show_img.shape = [28, 28]
    plt.subplot(4, 8, _ + 1)
    plt.imshow(show_img, cmap='gray')
    plt.axis('off')
# 打印第一层输出
layer_1 = K.function([model.layers[0].input], [model.layers[1].output])
f1 = layer_1([test_data])[0]
print(f1.shape)

#第一层卷积块第二次卷积后的特征图展示，输出是（1,28,28,32）
for _ in range(32):
    show_img = f1[:, :, :, _]
    show_img.shape = [28, 28]
    plt.subplot(4, 8, _ + 1)
    plt.imshow(show_img, cmap='gray')
    plt.axis('off')
layer_2 = K.function([model.layers[0].input], [model.layers[2].output])
f2 = layer_2([test_data])[0]
print(f2.shape)

#第一层卷积块池化后的特征图展示，输出是（1,14,14,32）
for _ in range(32):
    show_img = f2[:, :, :, _]
    show_img.shape = [14, 14]
    plt.subplot(4, 8, _ + 1)
    plt.imshow(show_img, cmap='gray')
    plt.axis('off')
layer_3 = K.function([model.layers[0].input], [model.layers[4].output])
f3 = layer_3([test_data])[0]
print(f3.shape)

#第二层卷积块第一次卷积后的特征图展示，输出是（1,14,14,64）
for _ in range(64):
    show_img = f3[:, :, :, _]
    show_img.shape = [14, 14]
    plt.subplot(8, 8, _ + 1)
    plt.imshow(show_img, cmap='gray')
    plt.axis('off')
layer_5 = K.function([model.layers[0].input], [model.layers[5].output])
f5 = layer_5([test_data])[0]
print(f5.shape)

#第二层卷积块第二次卷积后的特征图展示，输出是（1,14,14,64）
for _ in range(64):
    show_img = f5[:, :, :, _]
    show_img.shape = [14, 14]
    plt.subplot(8, 8, _ + 1)
    plt.imshow(show_img, cmap='gray')
    plt.axis('off')
layer_6 = K.function([model.layers[0].input], [model.layers[6].output])
f6 = layer_6([test_data])[0]
print(f6.shape)

#第二层卷积块池化后的特征图展示，输出是（1,7,7,64）
for _ in range(64):
    show_img = f6[:, :, :, _]
    show_img.shape = [7, 7]
    plt.subplot(8, 8, _ + 1)
    plt.imshow(show_img, cmap='gray')
    plt.axis('off')