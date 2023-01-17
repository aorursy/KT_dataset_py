#!user/python/bin

#-*- coding: utf-8 -*-





## 整体上调整efficientnetB7   验证集不使用数据增强版本  应用于6分类的垃圾分类

## 移植必须改变的部分有  目录 图像文件名和标签名部分

## 若二分类 则需要改变激活函数等内容



!pip install -q efficientnet

from efficientnet.tfkeras import EfficientNetB7

height , width = 224 , 224

conv_base = EfficientNetB7(weights='noisy-student',

                  include_top=False,

                  input_shape=(height,width,3))

conv_base.summary()



last_layer_shape = (7,7,2560)

input_length = last_layer_shape[0] * last_layer_shape[1] * last_layer_shape[2]





##使用预训练的卷积基提取特征

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

import pandas as pd

import os ,shutil



##图像目录

dir_train = '../input/rubbish/train'

dir_test = '../input/rubbish/test'

## 图像文件名 和 标签

dataframe_train = pd.read_csv('../input/rubbish/train.csv')

dataframe_test = pd.DataFrame({ 'filename':['{}.jpg'.format(i) for i in range(506)] })

## dataframe 中图像文件名 和 标签 对应的列名

x_col = 'filename'

y_col = 'label'

##求类别个数 和 类别字典

classes = list(set(dataframe_train.iloc[:,1]))  #所有类别

classes_num = len(classes)



batch_size = 16  #16

from keras import models

from keras import layers



model = models.Sequential()

model.add(conv_base)  #在卷积基的基础上添加分类器

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(classes_num,activation='softmax'))

# model.summary()

conv_base.trainable = True



##使用学习率较低的Adam优化器

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers



train_datagen = ImageDataGenerator(rescale=1./255,

                                   rotation_range=30,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   channel_shift_range=10,

                                   horizontal_flip=True,

                                   fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1./255)   #注意  不能增强验证数据



split_ind = int(dataframe_train.shape[0]*0.8)     #划分训练集和验证集的分隔下标

train_generator = train_datagen.flow_from_dataframe(dataframe_train[:split_ind],        #用train_datagen 有数据增强

                                                    dir_train,

                                                    x_col,y_col,

                                                    target_size=(height,width),

                                                    batch_size=batch_size,

                                                    class_mode='categorical',

                                                    # seed = 666,

                                                    classes=classes)

validation_generator = test_datagen.flow_from_dataframe(dataframe_train[split_ind:],   # 用test_datagen 没数据增强

                                                    dir_train,

                                                    x_col,y_col,

                                                    target_size=(height,width),

                                                    batch_size=batch_size,

                                                    class_mode='categorical',

                                                    # seed=666,

                                                    classes=classes)

test_generator = test_datagen.flow_from_dataframe(dataframe_test,

                                                  dir_test,

                                                  x_col,None,

                                                  batch_size=batch_size,

                                                  target_size=(height,width),

                                                  class_mode=None,       #用于预测的数据必须 class_mode=None 和 y_col = None

                                                  shuffle=False)         #并且shuffle=False

model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.Adam(lr=1e-5),

              metrics=['acc'])

history = model.fit_generator(train_generator,

                              epochs=300,   # 50

                              validation_data=validation_generator)

##绘制结果

import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')

plt.plot(epochs,val_acc,'b',label='Validation acc')

plt.title('Training and Validationg accuracy')

plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')

plt.plot(epochs,val_loss,'b',label='Validation loss')

plt.title('Training and Validation loss')

plt.legend()

plt.show()

##评估模型

val_loss,val_acc = model.evaluate(validation_generator)

print('val acc:',val_acc)



##应用在新图片

test_generator.reset()  # predict_generator之前必须reset()

# plt.imshow(next(test_generator)[0])

pred = model.predict_generator(test_generator)

predicted_class_indices=np.argmax(pred,axis=1)

labels = train_generator.class_indices

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames   #返回测试集图像的文件名序列

results=pd.DataFrame({"filename":filenames, "label":predictions})

results.to_csv("results.csv",index=False)

model.save('model_efnB7_rubbish_classify.h5')