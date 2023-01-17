from IPython.core.interactiveshell import InteractiveShell  #执行该代码可以使得当前nb支持多输出
InteractiveShell.ast_node_interactivity = "all" 
import sys
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
#读入训练以及测试数据
test_data=pd.read_csv('C:/Users/Administrator/python演示程序/Dight Recognizer/data/digit-recognizer (1)/test.csv')
test_data.head()
train_data=pd.read_csv('C:/Users/Administrator/python演示程序/Dight Recognizer/data/digit-recognizer (1)/train.csv')
train_data.head()
#读入测试数据，并reshape成28000个28*28的array
def reading_test_data():
    x_test_data=test_data.values[:,:]
    x_test_data.shape
    x_test_data=x_test_data.reshape(x_test_data.shape[0],28,28,1)
    return x_test_data
# reading_test_data()
#读入训练数据，将第一行的label分开，并对data reshape 成42000个28*28的array
def reading_train_data():
    #从第二列开始读入数据
    x_train_data=train_data.values[:,1:]
    #第一行为label
    x_train_label=train_data.values[:,0]
    x_train_data.shape
    x_train_data=x_train_data.reshape(x_train_data.shape[0],28,28,1)
    #将数据归一化
    x_train_data=x_train_data/255
    
    
    fix_label = np.zeros((x_train_label.shape[0], 10))
    for i in range(x_train_label.shape[0]):
        fix_label[i][x_train_label[i]] = 1
    return x_train_data,fix_label
# reading_train_data()

x_train_data,fix_label=reading_train_data()

#模仿LeNet进行模型建立
def model(x_train_data,fix_label):
    #序贯模型
    model=Sequential()
    #加入卷积层，数据长宽为28，28；卷积核长宽3，32个filter，padding方式为same，激活函数为reLu
    model.add(Conv2D(input_shape=(28,28,1),kernel_size=(3,3),filters=32,padding='same',activation='relu'))
    #加入池化层，用最大值进行池化，池化范围为2，2，步长为2，2
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    #第二次卷积，数据长宽为14，14；卷积核长宽3，32个filter，padding方式为same，激活函数为reLu
    model.add(Conv2D(kernel_size=(3,3),filters=32,padding='same',activation='relu'))
    #再次池化，用最大值进行池化，池化范围为2，2，步长为2，2
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    #将数据展平
    model.add(Flatten())
    #加入全连接层
    model.add(Dense(128,activation='relu'))
    #将输入映射到0-1之间（神经元权重）
    model.add(Dense(10,activation='softmax'))
    #输出每层的参数情况
    model.summary()
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    model.fit(x_train_data,fix_label, batch_size=3000, epochs=30,verbose=1, validation_split=0.2)

    
model(x_train_data,fix_label)
score = model.evaluate(x_train_data, fix_label, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
