# 加载训练数据和测试数据

import numpy as np

pokemon_train = np.load("/kaggle/input/youthai-competition/pokemon_train.npy")

pokemon_test = np.load("/kaggle/input/youthai-competition/pokemon_test.npy")
# 训练数据的第一列是标签，后面128*128*3列是图片每一个像素

# 测试数据没有标签

x_train = pokemon_train[:, 1:].reshape(-1, 128, 128, 3)

y_train = pokemon_train[:, 0].reshape([-1])

x_test = pokemon_test.reshape(-1, 128, 128, 3)



# 可视化前10个训练数据

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(10, 4))

axes = axes.flatten()

for i in range(10):

    axes[i].imshow(x_train[i])

    axes[i].set_xticks([])

    axes[i].set_yticks([])

plt.tight_layout()

plt.show()



print('这十张图片的标签分别是：', y_train[:10])



# 将标签对应为宝可梦种类

label_name = {0:'妙蛙种子', 1:'小火龙', 2:'超梦', 3:'皮卡丘', 4:'杰尼龟'}

name_list = []

for i in range(10):

    name_list.append(label_name[y_train[i]])

print('这十张图片标签对应的宝可梦种类分别为：', name_list)
# 开始训练一个简单的CNN模型

import keras

from keras.models import Sequential

from keras.layers import Dense, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import RMSprop

from keras.layers import Dropout



x_train = pokemon_train[:, 1:].reshape(-1, 128, 128, 3)

y_train = pokemon_train[:, 0].reshape([-1])

#x_test = pokemon_test.reshape(-1, 128, 128, 3)



x_train = x_train / 255

y_train = keras.utils.to_categorical(y_train)

x_test = x_train[900:]

x_train = x_train[:900]

y_test = y_train[900:]

y_train = y_train[:900]



model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)))

model.add(Conv2D(32,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64,(5,5),activation='relu'))

model.add(Conv2D(64,(5,5),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation="relu"))

model.add(Dense(5, activation="softmax"))



model.compile(loss="categorical_crossentropy",  optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])



model.fit(x_train, y_train, batch_size=16, epochs=5)
score = model.evaluate(x_test,y_test,verbose=0)

print('loss:',score[0])

print('accuracy:',score[1])