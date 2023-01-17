import pandas as pd

diabetes = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
print (diabetes)
import numpy as np

def normalize(X):    #归一化

    mean=np.mean(X)

    std=np.std(X)

    X=(X-mean)/std

    return X
X=normalize(diabetes.iloc[:,0:8])

Y=diabetes.iloc[:,8] #Y是无需归一化的输出
import keras

from keras import layers

#模型构建

model=keras.Sequential()

model.add(layers.Dense(16,input_dim=8,activation='relu')) #输入层为8个输入，同时安排16个神经元的第一隐藏层 #relu为常用激活函数

model.add(layers.Dense(8,activation='relu'))              #有8个神经元的第二隐藏层

model.add(layers.Dense(1,activation='sigmoid'))           #输出层 #Sigmoid 为输出层首选激活函数



#模型编辑

model.compile(loss='binary_crossentropy', #采用二进制交叉熵损失函数

              optimizer='adam',           #使用梯度下降算法Adam优化器

              metrics=['accuracy']        #采用分类准确度作为度量模型的标准

             )
#定义一些超参数 （形成习惯）

epochs=150 #迭代次数

batch_size=10 #单个批次数据个数



#训练模型

history=model.fit(x=X,y=Y,epochs=epochs,batch_size=batch_size)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.title('Loss')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.show()

plt.plot(history.history['accuracy'])

plt.title('Accuracy')

plt.xlabel('epoch')

plt.ylabel('Accuracy')

plt.show()
scores=model.evaluate(x=X,y=Y)

print("损失值;",scores[0],'\n','准确度:',scores[1]*100,'%')
#······