# 加载训练数据和测试数据

import numpy as np

pokemon_train = np.load("../input/pokemon_train.npy")

pokemon_test = np.load("../input/pokemon_test.npy")
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

from keras.optimizers import Adam



x_train = pokemon_train[:, 1:].reshape(-1, 128, 128, 3)

y_train = pokemon_train[:, 0].reshape([-1])

x_test = pokemon_test.reshape(-1, 128, 128, 3)



x_train = x_train / 255

y_train = keras.utils.to_categorical(y_train)



model = Sequential()

model.add(Conv2D(32, (5, 5), activation="relu", input_shape=(128, 128, 3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256, activation="relu"))

model.add(Dense(5, activation="softmax"))



model.compile(loss="categorical_crossentropy",  optimizer=Adam(lr=0.00001), metrics=['accuracy'])



model.fit(x_train, y_train, batch_size=32, epochs=8)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# 用训练好的模型来预测测试集的标签

x_test = pokemon_test.reshape(-1, 128, 128, 3)

x_test = x_test / 255

predict_labels = model.predict_classes(x_test, batch_size=32)



predict_label_csv = np.hstack([(np.arange(predict_labels.shape[0])+1).reshape([-1, 1]), predict_labels.reshape([-1, 1])])

np.savetxt('predict_label.csv', predict_label_csv, delimiter = ',', header='Id,Category')



# 为预测的标签生成一个下载链接，下载得到的csv文件可以直接提交然后查看自己的排名

def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



create_download_link(filename='predict_label.csv')