# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Softmax
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
train_file = '../input/train.csv'
test_file = '../input/test.csv'
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

inputs = Input(shape=[28,28,1])
net = Conv2D(6, kernel_size=[3,3] , strides=[1,1], activation='relu', padding='same')(inputs)
net = MaxPooling2D((2,2), strides=(2,2), padding="valid")(net)
print(net.shape)
net
net = Conv2D(16, kernel_size=[5,5], strides=[1,1], activation='relu', padding='valid')(net)
net = MaxPooling2D()(net)
print(net.shape)
# net = Conv2D(64, kernel_size=[3,3] , strides=[1,1], activation='relu', padding='valid')(net)
# net = MaxPooling2D()(net)
# print(net.shape)
net = Conv2D(120, kernel_size=[5,5], strides=[1,1], activation='relu', padding='valid')(net)
net = Flatten()(net)
net = Dropout(rate=0.5)(net)
net = Dense(84, activation='relu')(net)
net = Dropout(rate=0.5)(net)
outputs = Dense(10, activation='softmax')(net)
learn_rate = 0.001
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(lr=learn_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
epochs = 20
batch_size = 128
x_all_train =  np.array(train.drop(['label'], axis=1)).reshape((-1,28,28, 1))
y_all_train = pd.get_dummies(train['label'])
for i in range(epochs):
    num = int(train.shape[0]/batch_size)
    for j in range(num):
        train_batch = train.sample(n=batch_size)
        x_train = np.array(train_batch.drop(['label'], axis=1)).reshape((-1,28,28, 1))
        y_train = pd.get_dummies(train_batch['label'])
        loss_and_metrics = model.train_on_batch(x_train, y_train)  # 开始训练
        
        if (j+1) % 100 == 0:
            print("%dth_epoch: %d/%d\nloss: %f\nacuraccy:%f"% (i, j+1, batch_size, loss_and_metrics[0], loss_and_metrics[1]))
#     loss_and_metrics = model.evaluate(x_train, y_train)
    loss_and_metrics = model.evaluate(x_all_train, y_all_train)
    print("epoch: %d/%d\nloss: %f\nacuraccy:%f\n"% (i, num, loss_and_metrics[0], loss_and_metrics[1]))

x_test = np.array(test).reshape((-1,28,28,1))
y_hat = model.predict(x_test)
y_pred = np.argmax(y_hat, 1)
res = pd.Series(y_pred, name="Label")
imgid = pd.Series(range(1, 28001), name="ImageId")
submission = pd.concat([imgid, res], axis=1)
print(submission[:10])
submission.to_csv("mnist_sub.csv",index=False)
