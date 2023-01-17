# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from subprocess import check_output
print(check_output(["ls","../input/fashionmnist"]).decode("utf8"))
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
data_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

print(input_shape)
print()
print(data_train.shape)
print()
print(data_test.shape)
#在数据集df中用.iloc将标签和特征切片
X=np.array(data_train.iloc[:,1:])
y_=np.array(data_train.iloc[:,0 ])
print(X)
print()
print(y_)

#用to_categorical转换one-hot
y=to_categorical(y_)
print(y)
#在训练集中划分验证集
print(X.shape)
print()
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,
                                             random_state=13)
print(X_train.shape)
print()
#另一种划分,如果修改随机数种子random_state会造成划分结果的内容的不同（随机切分行）
X_train2,X_val2,y_train2,y_val2=train_test_split(X,y,test_size=0.3,
                                             random_state=13)
print(X_train2.shape)
print()
#Test data
X_test=np.array(data_test.iloc[:,1:])
y_test=to_categorical(np.array(data_test.iloc[:,0]))
print(X_test)
print()
print(y_test)
print()
print(X_test.shape)
print()
print(y_test.shape)
#对数据进行reshape,原本一行784个数据像素顺序排列，现在依照28*28排列到新矩阵的2,3维度
X_train=X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
X_test=X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
X_val=X_val.reshape(X_val.shape[0],img_rows,img_cols,1)

print(X_train.shape)
#数据标准定为float32
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_val=X_val.astype('float32')
print(X_train[1][1][1][0])
print(type(X_train[1][1][1][0]))
print()
#归一化（maxmin）
X_train/=255
X_test/=255
X_val/=255
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization

#一次送入256batch的数据
#分10类
#50个迭代次数，一个epochs表示将训练集中所有的样本完整过一遍
batch_size=256
num_classes=10
epochs=50

img_rows,img_cols=28,28

#调用keras的顺序模型model，并且利用model.add进行模型的逐层构建
#下列结构是按照CNN卷积网络结构顺序构建
model=Sequential()
#第一层卷积层
#第一个参数32，filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
#激活函数用relu，初始化权值方式是“HE正态分布”
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=input_shape))
#2Dmaxpooling过程，在图像中2*2的范围取最大
model.add(MaxPooling2D((2,2)))
#dropout层能让通过信号一定概率停止（0.25）
model.add(Dropout(0.25))
#再加一层过滤器，输出64个特征
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(Dropout(0.4))
#类似压缩，flatten可以让多维的变得扁平（flat）
model.add(Flatten())
#dense全连接层，128个filter，用relu激活函数过滤
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.3))
#最后一层全连接按照标签数据来输出，用softmax作为激活函数
model.add(Dense(num_classes,activation='softmax'))
#compile完成模型的编译，使用categorical_crossentropy交叉熵，优化器基于Adam算法（一种优化损失函数的自适应学习率的算法）
model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adam(),
             metrics=['accuracy'])#用准确率作为评估好坏的指标
model.summary()

#训练
#用fit方法拟合，输入训练集，和初始化参数
history=model.fit(X_train,y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,
                 validation_data=(X_val,y_val))
#利用keras的evaluate接口，用测试集进行最终的性能评估
#evaluate:Returns the loss value & metrics values for the model in test mode.
score=model.evaluate(X_test,y_test,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#新老版本的keras对history内置对象名称的定义不同
for key,values in  dict.items(history.history):
    print (key)
import matplotlib.pyplot as plt#导入plt绘制loss
%matplotlib inline
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
predicted_classes = model.predict_classes(X_test)
y_true = data_test.iloc[:, 0]
correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]
from sklearn.metrics import classification_report#调用classification_report打印模型性能报告
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_true, predicted_classes, target_names=target_names))