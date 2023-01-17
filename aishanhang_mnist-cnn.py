#导入所需包
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import seaborn as sns
%matplotlib inline
np.random.seed(2)#设置随机数种子

#导入模型
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
import itertools
print(os.listdir("../input"))
sns.set(style='white',context='notebook',palette='deep')
#数据准备
train=pd.read_csv('../input/train.csv')#读入数据
test=pd.read_csv('../input/test.csv')
Y_train=train['label']#分离标签
X_train=train.drop('label',axis=1)
del train
g=sns.countplot(Y_train)#展示标签情况
Y_train.value_counts()#统计标签情况
#查看数据缺失情况
X_train.isnull().any().describe()
test.isnull().any().describe()
#正则化数据，减少光线的影响，同时加快计算速度
X_train=X_train/255.0
test=test/255.0
#调整数据格式，784*1转换成28*28*1
X_train=X_train.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)

#标签编码，one-hot形式
Y_train=to_categorical(Y_train,num_classes=10)
#生成训练数据和验证数据集
random_seed=2
X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.1,random_state=random_seed)
#展示数据,keyerror
g=plt.imshow(X_train[0][:,:,0])
#定义CNN模型
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
#设置优化器
optimizer=RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)

#编译模型
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
#设置学习率退化器,动态改变学习率
learning_rate_reduction=ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5,min_lr=0.00001)
epochs=28#设置为30:99.514%
batch_size=86
#数据增强
datagen=ImageDataGenerator(
    featurewise_center=False,#设置输入均值为0在整个数据集上
    samplewise_center=False,#设置每个样本均值为0
    featurewise_std_normalization=False,#按照数据集的std分割输入
    samplewise_std_normalization=False,#按照每个样本的std分割输入
    zca_whitening=False,#应用ZCA漂白
    rotation_range=10,#旋转度数[0,180]
    zoom_range=0.1,#随机放大图片
    width_shift_range=0.1,#水平方向随机移动（以整个图像宽度）
    height_shift_range=0.1,#垂直方向随机移动
    horizontal_flip=False,#随机水平翻动
    vertical_flip=False#随机垂直翻动
                          )
datagen.fit(X_train)
# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])
#训练模型
#history=model.fit_generator(datagen.flow(X_train,Y_train,batch_size=batch_size),
#                            epochs=epochs,validation_data=(X_val,Y_val),
#                            verbose=2,steps_per_epoch=X_train.shape[0],
#                           callbacks=[learning_rate_reduction])

#评估模型
#训练和验证曲线
fig,ax=plt.subplots(2,1)
ax[0].plot(history.history['loss'],color='b',label='Training loss')
ax[0].plot(history.history['val_loss'],color= 'r',label='validation loss',axes=ax[0])
legend=ax[0].legend(loc='best',shadow=True)

ax[1].plot(history.history['acc'],color='b',label='Training accuracy')
ax[1].plot(history.history['val_acc'],color='r',label='Validation accuracy')
legend=ax[1].legend(loc='best',shadow=True)
#混淆函数
def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    
    if(normalize):
        cm=cm.astype('float')/cm.sum(axis=1)[:,np/newaxis]
        
    thresh=cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment='center',color='white' if cm[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

Y_pred=model.predict(X_val)#预测值
Y_pre_classes=np.argmax(Y_pred,axis=1)#转成one-hot向量
Y_true=np.argmax(Y_val,axis=1)#真实值one-hot向量
confusion_mtx=confusion_matrix(Y_true,Y_pre_classes)#生成混淆矩阵
plot_confusion_matrix(confusion_mtx,classes=range(10))

#预测结果
results=model.predict(test)
results=np.argmax(results,axis=1)
results=pd.Series(results,name='Label')

submission=pd.concat([pd.Series(range(1,28001),name='ImageId'),results],axis=1)
submission.to_csv('result.csv',index=False)