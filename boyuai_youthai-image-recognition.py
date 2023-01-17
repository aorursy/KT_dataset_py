import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



from matplotlib import pyplot as plt
mnist_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

mnist_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

x_train = np.array(mnist_train.iloc[:, 1:]).reshape(-1, 28, 28)

y_train = np.array(mnist_train.iloc[:, 0])

x_test = np.array(mnist_test.iloc[:, 1:]).reshape(-1, 28, 28)

y_test = np.array(mnist_test.iloc[:, 0])
fig, axes = plt.subplots(2, 5, figsize=(10, 4)) # 新建一个包含10张子图2行5列的画布

axes = axes.flatten() # axes中存储了每一个子图

for i in range(10): # 循环10次（画10张图）

    axes[i].imshow(x_train[i], cmap="gray_r") # 将x_train的第i张图画在第i个子图上，这里我们用cmap="gray_r"即反灰度图，数字越大颜色越黑，数字越小颜色越白 

    axes[i].set_xticks([]) # 移除图像的x轴刻度

    axes[i].set_yticks([]) # 移除图像的y轴刻度

plt.tight_layout() # 采用更美观的布局方式

plt.show() # 显示图片
fig, axes = plt.subplots(10, 10, figsize=(20, 20)) # 新建一个包含100张子图的10行10列的画布

for i in range(10): # 对于每一个数字i

    indice = np.where(y_train == i)[0] #找到标签为数字i的图像下标

    for j in range(10): # 输出前10张图片

        axes[i][j].imshow(x_train[indice[j]], cmap="gray_r")

        axes[i][j].set_xticks([])

        axes[i][j].set_yticks([])

plt.tight_layout()

plt.show()
from os import listdir, makedirs

from os.path import join, exists, expanduser



cache_dir = expanduser(join('~', '.keras'))

if not exists(cache_dir):

    makedirs(cache_dir)

datasets_dir = join(cache_dir, 'datasets') # /cifar-10-batches-py

if not exists(datasets_dir):

    makedirs(datasets_dir)

    

!cp ../input/cifar10-python/cifar-10-python.tar.gz ~/.keras/datasets/

!ln -s  ~/.keras/datasets/cifar-10-python.tar.gz ~/.keras/datasets/cifar-10-batches-py.tar.gz

!tar xzvf ~/.keras/datasets/cifar-10-python.tar.gz -C ~/.keras/datasets/
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
fig, axes = plt.subplots(2, 5, figsize=(10, 4)) # 新建一个包含10张子图2行5列的画布

axes = axes.flatten() # axes中存储了每一个子图

for i in range(10): # 循环10次（画10张图）

    axes[i].imshow(x_train[i]) # 将x_train的第i张图画在第i个子图上

    axes[i].set_xticks([]) # 移除图像的x轴刻度

    axes[i].set_yticks([]) # 移除图像的y轴刻度

plt.tight_layout() # 采用更美观的布局方式

plt.show() # 显示图片
fig, axes = plt.subplots(10, 10, figsize=(20, 20)) # 新建一个包含100张子图的10行10列的画布

for i in range(10): # 对于每一类物体

    indice = np.where(y_train == i)[0] #找到标签为i的图像下标

    for j in range(10): # 输出前10张图片

        axes[i][j].imshow(x_train[indice[j]], cmap="gray_r")

        axes[i][j].set_xticks([])

        axes[i][j].set_yticks([])

plt.tight_layout()

plt.show()
from sklearn.neighbors import KNeighborsClassifier

from keras.datasets import mnist

from matplotlib import pyplot as plt

import numpy as np
mnist_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

mnist_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

x_train = np.array(mnist_train.iloc[:, 1:]).reshape(-1, 28, 28)

y_train = np.array(mnist_train.iloc[:, 0])

x_test = np.array(mnist_test.iloc[:, 1:]).reshape(-1, 28, 28)

y_test = np.array(mnist_test.iloc[:, 0])
n_train = x_train.shape[0] # 训练数据数量

n_test = x_test.shape[0] # 测试数据数量

print("原输入数据的形状")

print(x_train.shape)

print(x_test.shape)



# 使用reshape方法将图像展开成向量

x_train = x_train.reshape(n_train, -1) 

x_test = x_test.reshape(n_test, -1)

print("reshape后输入数据的数据形状")

print(x_train.shape)

print(x_test.shape)
k = 5

knc = KNeighborsClassifier(n_neighbors=k)
knc.fit(x_train, y_train)
y_predict = knc.predict(x_test)
accuracy = np.sum(y_predict == y_test) / n_test

print("准确度为 %f" % accuracy)
indice = np.random.choice(np.where(y_predict != y_test)[0], size=10) # 随机选择10张分类错误的图像

fig, axes = plt.subplots(2, 5, figsize=(10, 4))

axes = axes.flatten()

for i, idx in enumerate(indice):

    axes[i].imshow(x_test[idx].reshape(28, 28), cmap="gray_r")

    axes[i].set_xticks([])

    axes[i].set_yticks([])

    axes[i].set_title("y_predict: %d\ny_test: %d" % (y_predict[idx], y_test[idx]))

plt.tight_layout()

plt.show()
import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Flatten

from keras.layers import Conv2D, MaxPooling2D
mnist_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

mnist_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

x_train = np.array(mnist_train.iloc[:, 1:]).reshape(-1, 28, 28)

y_train = np.array(mnist_train.iloc[:, 0])

x_test = np.array(mnist_test.iloc[:, 1:]).reshape(-1, 28, 28)

y_test = np.array(mnist_test.iloc[:, 0])
width, height = x_train.shape[1], x_train.shape[2]

n_train = x_train.shape[0]

n_test = x_test.shape[0]



x_train = x_train.reshape(n_train, width, height, 1)

x_test = x_test.reshape(n_test, width, height, 1)

print("reshape后的输入形状")

print(x_train.shape)

print(x_test.shape)



y_train = keras.utils.to_categorical(y_train)

y_test = keras.utils.to_categorical(y_test)

print("独热化后的输出形状")

print(y_train.shape)

print(y_test.shape)



print("处理前的最大值为%f" % x_train.max())

x_train = x_train / 255

x_test = x_test / 255

print("处理后的最大值为%f" % x_train.max())
model = Sequential()

model.add(Conv2D(32, (5, 5), activation="relu", input_shape=(width, height, 1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256, activation="relu"))

model.add(Dense(10, activation="softmax"))
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test)

print("损失为%f" % score[0])

print("准确度为%F" % score[1])
from keras.datasets import cifar10

from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
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
y_train = keras.utils.to_categorical(y_train)

y_test = keras.utils.to_categorical(y_test)

x_train = x_train / 255

x_test = x_test / 255
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