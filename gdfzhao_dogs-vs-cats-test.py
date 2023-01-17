# %% [code]

from keras.models import Sequential

from keras.applications import ResNet50, Xception, InceptionV3  #导入预训练模型

from keras.layers import *

from keras import optimizers

from sklearn.model_selection import train_test_split

import os

import pandas as pd

from keras.preprocessing import image

import matplotlib.pyplot as plt

import keras.backend as K

import numpy as np





#参数设定

Fast_Run = False

image_width = 128

image_height = 128

image_size = (image_width, image_height)

image_channels = 3

batch_size = 50

#准备数据

filenames = os.listdir('../input/dogs-vs-cats/train/train')

categories = []

for filename in filenames:

    category =filename.split('.')[0]

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})



df['category'] = df['category'].replace({0: 'cat', 1: 'dog'})

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)     #重置随机排列后的序列索引

validate_df = validate_df.reset_index(drop=True)



total_train = train_df.shape[0]

total_validate = validate_df.shape[0]



train_datagen = image.ImageDataGenerator(     #图片生成器，数据进行增强，扩充数据集大小，增强模型的泛化能力

    rotation_range=15,             #旋转角度范围

    rescale=1./255,                #每个像素乘上缩放因子

    shear_range=0.1,               #透视变换范围

    zoom_range=0.2,                #缩放变换范围

    horizontal_flip=True,          #水平翻转

    width_shift_range=0.1,         #水平平移范围

    height_shift_range=0.1,

)



train_generator = train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/dogs-vs-cats/train/train",

    x_col='filename',

    y_col='category',

    target_size=(image_width, image_height),

    class_mode='categorical',   #生成2D的onehot标签

    batch_size=batch_size,

)



from keras.callbacks import EarlyStopping, ReduceLROnPlateau #防止过拟合提前停止，调整学习率

earlystop = EarlyStopping(patience=10)

leraning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=1e-5)

#学习率调整（监控量，两次，verbose=True则每次更新向外输出一条信息，factor学习率降低因子

callbacks = [earlystop, leraning_rate_reduction]



#构建模型

K.set_learning_phase(0)  #关闭resnet的BN层

conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(image_width, image_height, image_channels))

conv_base.trainable = False  #冻结卷积基，不改变权重

K.set_learning_phase(1)

model = Sequential()

model.add(conv_base)

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dense(2, activation='softmax'))    #二分类



RMSprop = optimizers.RMSprop(lr=1e-4)

model.compile(loss='categorical_crossentropy', optimizer=RMSprop, metrics=['accuracy'])

model.summary()



#callbacks

from keras.callbacks import EarlyStopping, ReduceLROnPlateau #防止过拟合提前停止，调整学习率

earlystop = EarlyStopping(patience=10)

leraning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=1e-5)

#学习率调整（监控量，两次，verbose=True则每次更新向外输出一条信息，factor学习率降低因子

callbacks = [earlystop, leraning_rate_reduction]



model.load_weights('/kaggle/working/model.h5')

#testing data

test_filename = os.listdir("../input/dogs-vs-cats/test/test")

test_df = pd.DataFrame({

    'filename': test_filename,

})

num_samples = test_df.shape[0]



test_datagen = image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(

    dataframe=test_df,

    directory="../input/dogs-vs-cats/test/test",

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=image_size,

    batch_size=batch_size,

    shuffle=False

)



#predict

predict = model.predict_generator(test_generator, steps=num_samples/batch_size) #将test迭代器送入，返回预测结果（Onehot)

print(predict)

test_df['category'] = np.argmax(predict, axis=-1)

#onehot 标签，表示概率，找到最大值索引

#获取模型的默认labels，生成字典

label_map = dict((value, key) for key, value in train_generator.class_indices.items())

print(label_map)

#class_indices.items返回可遍历的标签键值对dict_items([('cat', 0), ('dog', 1)])

test_df['category'] = test_df['category'].replace(label_map)   # 猫代替0，狗代替1

test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })



#submission

submission_df = test_df.copy() #浅拷贝，原对象不变

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('/kaggle/working/submission_df', index=False)
