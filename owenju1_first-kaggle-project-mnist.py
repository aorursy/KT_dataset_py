# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

raw_train = pd.read_csv('../input/train.csv')

raw_test = pd.read_csv('../input/test.csv')
raw_train.head()


# 将原始数据分为 labels 和 inputs

inputs, labels = raw_train.values[:, 1:], raw_train.values[:, 0:1]



# 测试数据没有标记label

tests = raw_test.values



# 打乱数据，抽取部分数据用作 validation 

import random

all_idx = np.arange(inputs.shape[0])

random.shuffle(all_idx)

train_inputs, train_labels = inputs[all_idx[5000:]], labels[all_idx[5000:]]

valid_inputs, valid_labels = inputs[all_idx[0:5000]],labels[all_idx[0:5000]]



# 对标签进行 one hot 处理

from keras.utils import to_categorical

one_hot_train_labels = to_categorical(train_labels, num_classes=10)

one_hot_valid_labels = to_categorical(valid_labels, num_classes=10)
# 展示数据

import matplotlib.pyplot as plt

def view_sample(index):

    plt.imshow(train_inputs[index].reshape(28,28))

    plt.title(train_labels[index])

    plt.show()
view_sample(10)
from keras.models import Sequential

from keras.layers import Dense, Activation
model_sigmoid = Sequential()

model_sigmoid.add(Dense(1024, activation='sigmoid', input_dim=784))

model_sigmoid.add(Dense(10, activation='softmax'))

model_sigmoid.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy']

)
# 训练数据

model_sigmoid.fit(train_inputs, one_hot_train_labels, epochs=10, batch_size=256)
score = model_sigmoid.evaluate(valid_inputs, one_hot_valid_labels, batch_size=56)

print(score)
model_tanh = Sequential()

model_tanh.add(Dense(1024, activation='tanh', input_dim=784))

model_tanh.add(Dense(10, activation='softmax'))

model_tanh.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy']

)
model_tanh.fit(train_inputs, one_hot_train_labels, epochs=10, batch_size=256)

score = model_tanh.evaluate(valid_inputs, one_hot_valid_labels, batch_size=128)

print(score)
id = np.arange(1,28001)

pred_classes = model_sigmoid.predict_classes(tests)



submission = pd.DataFrame({

    "ImageId": id,

    "Label": pred_classes})



print(submission[0:10])



submission.to_csv('predictions.csv', index=False)