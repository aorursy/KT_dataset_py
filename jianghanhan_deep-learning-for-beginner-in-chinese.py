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
%matplotlib inline

import matplotlib.pyplot as plt

import os

import sys

from pathlib import Path

import numpy as np

import pandas as pd

import sklearn

import tensorflow as tf



from tensorflow import keras 



##查看版本状况

print(tf.__version__)

print(sklearn.__version__)

print(pd.__version__)

print(np.__version__)



##载入要用的数据 

train = '../input/10-monkey-species/training/training'

valid = '../input/10-monkey-species/validation/validation/'

label_file = '../input/10-monkey-species/monkey_labels.txt'

print(os.path.exists(train))

print(os.path.exists(valid))

print(os.path.exists(label_file))

#读取标签资料

labels = pd.read_csv(label_file)

print(labels)

##定义要用的参数

height = 128  ##图片高

width = 128   ##图片宽

channels = 3   ##代表RGB 三种通道

batch_size = 64  ##批量大小

num_classes = 10   ##总共十种猴子种类





##利用keras作图像上采样!!

train_datagen = keras.preprocessing.image.ImageDataGenerator(

 rescale = 1./255    ##归一化

,rotation_range = 40  

,width_shift_range = 0.2

,height_shift_range = 0.2

,shear_range = 0.2

,zoom_range = 0.2

,horizontal_flip = True

,fill_mode = 'nearest'

)



##以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据

train_generator = train_datagen.flow_from_directory(train

                                                  ,target_size = (height,width)

                                                  ,batch_size = batch_size

                                                  ,seed=7

                                                  ,shuffle = True

                                                  ,class_mode = 'categorical'

                                                   )

                                                   

##验证集上做相同的事                                                 

valid_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)





valid_generator = valid_datagen.flow_from_directory(valid

                                                  ,target_size = (height,width)

                                                  ,batch_size = batch_size

                                                  ,seed=7

                                                  ,shuffle = False

                                                  ,class_mode = 'categorical'                                                   

                                                    )

##查看有多少图片数据!! 

train_num = train_generator.samples

valid_num = valid_generator.samples

print(train_num,valid_num)

for i in range(2):

    x,y = train_generator.next()

    print(x.shape,y.shape)

    print(y)
##用keras.models模块搭建模型!! 



model = keras.models.Sequential([

     keras.layers.Conv2D(filters = 32 , kernel_size = 3,padding='same'

                        ,activation = 'relu',input_shape = [width,height,channels])

     ,keras.layers.Conv2D(filters = 32 , kernel_size = 3,padding='same'

                        ,activation = 'relu') 

     ,keras.layers.MaxPool2D(pool_size = 2)

    

     ,keras.layers.Conv2D(filters = 64 , kernel_size = 3,padding='same'

                        ,activation = 'relu')

     ,keras.layers.Conv2D(filters = 64 , kernel_size = 3,padding='same'

                        ,activation = 'relu') 

     ,keras.layers.MaxPool2D(pool_size = 2)

    

     ,keras.layers.Conv2D(filters = 128 , kernel_size = 3,padding='same'

                        ,activation = 'relu')

     ,keras.layers.Conv2D(filters = 128 , kernel_size = 3,padding='same'

                        ,activation = 'relu') 

     ,keras.layers.MaxPool2D(pool_size = 2)

    

     ,keras.layers.Flatten()

     ,keras.layers.Dense(128,activation='relu')

     ,keras.layers.Dense(num_classes,activation = 'softmax')

    

])



##定义模型训练!!

model.compile(loss = 'categorical_crossentropy'

             ,optimizer = 'adam',metrics = ['accuracy'])



##查看模型!!

model.summary()
##训练模型!!

epochs = 10 

history = model.fit_generator(train_generator

                             ,steps_per_epoch = train_num//batch_size

                             ,epochs = epochs

                             ,validation_data = valid_generator

                             ,validation_steps = valid_num//batch_size)
print(history.history.keys())

##自定学习曲线函数

def plot_learing_curve(history,label,epoch,min_value,max_value):

    data = {}

    data[label] = history.history[label]

    data['val_'+label] = history.history['val_'+label]

    pd.DataFrame(data).plot(figsize=(8,5))

    plt.grid(True)

    plt.axis([0,epochs,min_value,max_value])

    plt.show()

    

##画图    

plot_learing_curve(history,'loss',epochs,1.5,2.5)

plot_learing_curve(history,'accuracy',epochs,0,1)