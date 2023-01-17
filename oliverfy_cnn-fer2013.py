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


import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow.keras.models import Sequential,Model

from tensorflow.keras.layers import Conv2D,Dropout,Dense,Flatten,MaxPool2D

from tensorflow.python.keras.callbacks import TensorBoard

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split



train=pd.read_csv('../input/ml2019spring-hw3/train.csv')

test=pd.read_csv('../input/ml2019spring-hw3/test.csv')
class Data_clean():

    def __init__(self,train_df):

        self._train_df=train_df

        self._train_df['feature'] = self._train_df['feature'].map(lambda x : np.array(list(map(float, x.split()))))

        self._image_size=self._train_df['feature'][0].size

        self._image_shape=(int(np.sqrt(self._image_size)),int(np.sqrt(self._image_size)))

        self._dataNum=self._train_df.shape[0]

        self._feature = np.array(self._train_df.feature.map(lambda x: x.reshape(self._image_shape)).values.tolist())

        if 'label' in train_df.columns: 

            self._distribution=train_df['label'].value_counts().sort_values()



            self._label=self._train_df['label'].values

            self._labelNum=self._train_df['label'].nunique()

            self._onehot=pd.get_dummies(self._train_df['label']).values

    @property

    def distribution(self):

        return self._distribution

    

    @property

    def image_size(self):

        return self._image_size

    

    @property

    def image_shape(self):

        return self._image_shape

    

    @property

    def dataNum(self):

        return self._dataNum

    

    @property

    def feature(self):

        return self._feature

    

    @property

    def label(self):

        return self._label

 

    @property

    def labelNum(self):

        return self._labelNum

    

    @property

    def onehot(self):

        return self._onehot



def plot_images(images,cls_true,cls_pred=None):

    assert len(images) == len(cls_true) == 9

    

    # Create figure with 3x3 sub-plots.创建子图

    fig, axes = plt.subplots(3, 3)#axes子图轴域

    fig.subplots_adjust(hspace=0.3, wspace=0.3)#调整边距和子图的间距

    for i, ax in enumerate(axes.flat):#flat方法是将整个numpy对象转换成了1维对象

        # Get the i'th image and reshape the array.

        #image = images[i].reshape(img_shape)

        

        

        # Ensure the noisy pixel-values are between 0 and 1.

        #image = np.clip(image, 0.0, 1.0)

 

        # Plot image.

        ax.imshow(images[i],

                  cmap = 'gray',#颜色图谱（colormap), 默认绘制为RGB(A)颜色空间。

                  interpolation='nearest')#最近邻差值

        # Show true and predicted classes.



        if cls_pred is None:#还没有预测结果

            xlabel = "True: {0}".format(name[cls_true[i]])#字符串格式化

        else:

            xlabel = "True: {0}, Pred: {1}".format(name[cls_true[i]], name[cls_pred[i]])

        

        # Show the classes as the label on the x-axis.

        ax.set_xlabel(xlabel)#把xlabel在图中标注出来

        

        # Remove ticks from the plot.

        ax.set_xticks([])#设置坐标轴刻度

        ax.set_yticks([])

    

    # Ensure the plot is shown correctly with multiple plots

    # in a single Notebook cell.

    plt.show()

data=Data_clean(train)
X_train=data.feature/255

y_label=data.label

y_onehot=data.onehot

X_train=X_train.reshape([-1,48,48,1])
#split train set and valid set

random_seed=1

X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_onehot, test_size = 0.1, random_state=random_seed)
model1=Sequential()

model1.add(Conv2D(64,kernel_size=3,activation='relu',input_shape=[48,48,1],padding='same'))

model1.add(BatchNormalization(axis=-1))

model1.add(MaxPool2D(pool_size=2,strides=1))

model1.add(Dropout(0.3))

model1.add(Conv2D(128,kernel_size=3,activation='relu',padding='same'))

model1.add(BatchNormalization(axis=-1))

model1.add(MaxPool2D(pool_size=2,strides=1))

model1.add(Dropout(0.3))



model1.add(Conv2D(256,kernel_size=3,activation='relu',padding='same'))

model1.add(BatchNormalization(axis=-1))

model1.add(MaxPool2D(pool_size=2,strides=2))

model1.add(Dropout(0.4))



model1.add(Conv2D(512,kernel_size=3,activation='relu',padding='same'))

model1.add(BatchNormalization(axis=-1))

model1.add(MaxPool2D(pool_size=2,strides=2))

model1.add(Dropout(0.4))



model1.add(Flatten())

model1.add(Dense(1024,activation='relu'))

model1.add(Dropout(0.5))



model1.add(Dense(512,activation='relu'))

model1.add(Dropout(0.5))



model1.add(Dense(7,activation='softmax'))
model1.summary()
model1.compile(loss='categorical_crossentropy',

               optimizer='adam',

               metrics=['accuracy'])




datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = False, # Randomly zoom image 

        width_shift_range=False,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=False,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)



#without data augmentation

#history=model1.fit(X_train,y_onehot,batch_size=128,epochs=10,validation_split=0.2)

batch_size=128

epochs=200

# Fit the model

history = model1.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size

                              )
fig,ax=plt.subplots(2,1,figsize=(10,8))

ax[0].plot(history.history['loss'],label='Training loss')

ax[0].plot(history.history['val_loss'],label='validation loss',axes=ax[0])

ax[0].legend(loc=0,shadow=True)



ax[1].plot(history.history['accuracy'],label='Training acc')

ax[1].plot(history.history['val_accuracy'],label='validation acc')

ax[1].legend(loc=0,shadow=True)

plt.show()
test_data=Data_clean(test)

test=(test_data.feature/255).reshape([-1,48,48,1])
results=model1.predict(test)

labels=np.argmax(results,axis=1)

labels=pd.Series(labels,name='label')
len(labels)
submission=pd.concat([pd.Series(range(len(labels)),name='id'),labels],axis=1)

submission.to_csv("cnn_fer2013.csv",index=False)