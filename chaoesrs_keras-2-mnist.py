import pandas as pd

fashion_mnist_test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")

fashion_mnist_train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")

#看一下规模

print(fashion_mnist_train.shape,fashion_mnist_test.shape) 

print(fashion_mnist_test)
#标签对应的物品

Label={0:'T-shirt/top', 1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle boot'}
import keras

from keras import layers

X_train=fashion_mnist_train.iloc[:,1:]

X_test=fashion_mnist_test.iloc[:,1:]

Y_train=fashion_mnist_train.iloc[:,0]

Y_test=fashion_mnist_test.iloc[:,0]

print(X_test.shape)
import matplotlib.pyplot as plt

import numpy as np

def plot_images_labels_prediction(images,#图像列表

                                  labels,#标签列表

                                  index,#从第index个开始显示

                                  num=10): #缺省一次显示10幅

    fig=plt.gcf() #获取当前图表

    fig.set_size_inches(10,12) #一英寸等于2.54cm

    if num>25:

        num=25     #最多25幅图

    for i in range(0,num):

        ax=plt.subplot(5,5,i+1)

        ax.imshow(np.reshape(np.array(images.loc[index]),(28,28)),cmap='binary')#显示第index个图像

        title='label='+Label[labels.loc[index]]#构建该图上要显示的标题

        ax.set_title(title,fontsize=10)

        ax.set_xticks([])#不显示坐标轴

        ax.set_yticks([])

        index+=1

    plt.show()

plot_images_labels_prediction(X_test,

                              Y_test,

                              44,

                              10)
X_train=np.reshape(np.array(X_train),(60000,28,28,1))/255

X_test=np.reshape(np.array(X_test),(10000,28,28,1))/255
from keras.utils import np_utils

Y_train=np_utils.to_categorical(Y_train)

Y_test=np_utils.to_categorical(Y_test)

print(X_train.shape)
from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

model=keras.Sequential()

model.add(Conv2D(32,(5,5),input_shape=(28,28,1),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(layers.Dropout(0.2))

model.add(layers.Flatten())

model.add(layers.Dense(units=10,kernel_initializer='normal',activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=25,batch_size=200,verbose=2)
scores = model.evaluate(X_test, Y_test, verbose=0)
print(scores)