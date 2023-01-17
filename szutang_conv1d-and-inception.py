from google.colab import files

files.upload()
!mkdir -p ~/.kaggle 

!cp kaggle.json ~/.kaggle/ 

!chmod 600 ~/.kaggle/kaggle.json
import os

import glob

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras import Sequential, Model

from tensorflow.keras.layers import Conv1D, Multiply, MaxPool1D, RepeatVector, Reshape, Activation, MaxPool1D, GlobalAveragePooling1D, MaxPool2D, Concatenate, Add, Flatten, Conv2D, Input, Dense, Dropout, Activation, BatchNormalization

from tensorflow.keras.regularizers import l2

from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

from tensorflow.keras.models import Model

import warnings

warnings.filterwarnings('ignore')


test = pd.read_csv('test_clean.csv')

train = pd.read_csv('train_clean.csv')

from keras.utils.np_utils import to_categorical



X_train = train.signal.values

Y_train = train.open_channels.values

Y_train = to_categorical(Y_train, 11)
# window size: 100, [-80, +20]





def train_generator(X_train, Y_train, batch_size):

    while True:

        x_train = np.empty((0, 100, 1))

        y_train = np.empty((0,11))



        for i in range(batch_size):

            time = int(np.random.uniform(80, 5000000-20, 1))

            

            y_train = np.append(y_train, Y_train[time].reshape(1,11), axis=0)

            x_train = np.append(x_train, X_train[time-80  : time+20 : 1].reshape(1, -1, 1), axis=0)

        

        yield x_train, y_train        
def inception_block(inputs, filters):

    

    # 分支1

    conv_1 = Conv1D(filters=filters, kernel_size=1, strides=1, padding="same", activation="relu")(inputs)

    



    # 分支2

    conv_2 = Conv1D(filters=filters, kernel_size=3, strides=1, padding="same", activation="relu")(inputs)



    # 分支3

    conv_3 = Conv1D(filters=filters, kernel_size=3, strides=1, padding="same", activation="relu")(inputs)

    conv_3 = Conv1D(filters=filters, kernel_size=3, strides=1, padding="same", activation="relu")(conv_3)



    # 合并

    outputs = Concatenate(axis=-1)([conv_1, conv_2, conv_3])

    outputs = Conv1D(filters=filters, kernel_size=1, strides=1, padding="same", activation="relu")(outputs)



    return outputs



def se_block(inputs, k): #SE Block模块

   

    # 输入尺寸

    input_shape = K.int_shape(inputs)



    # 全局平均池化

    outputs = GlobalAveragePooling1D()(inputs)



    # 计算每个通道的重要性

    outputs = Dense(units=int(input_shape[-1] / k), activation="relu")(outputs)

    outputs = Dense(units=input_shape[-1], activation="sigmoid")(outputs)

    

    # 重新标定每个通道

    outputs = RepeatVector(input_shape[1] )(outputs)

    outputs = Reshape([input_shape[1], input_shape[2]])(outputs)

    outputs = Multiply()([inputs, outputs])

    

    return outputs



# inception maxpooling selection layer

def ims_layer(inputs, filters, pool_size):#特征提取层

    

    inception = inception_block(inputs, filters)

    pool = MaxPool1D(pool_size=pool_size, strides=pool_size, padding="same")(inception)

    se = se_block(pool, 4)

    

    return se 



def fc_layer(inputs, units):#最后的全连接层

    

    outputs = Dense(units=units, activation="relu")(inputs)

    outputs = Dropout(0.5)(outputs)

    

    return outputs



def model_build():

    # 原始输入数据

    raw_input = Input((100, 1))



    # ims_1

    ims_1 = ims_layer(raw_input, 64, 2) 

    

    # ims_2

    ims_2 = ims_layer(ims_1, 128, 3) 



    # ims_3

    ims_3 = ims_layer(ims_2, 256, 3)



    # ims_4

    ims_4 = ims_layer(ims_3, 256, 3)

    

    # Flatten

    flatten = Flatten()(ims_4)

    

    # 连接原始信号特征和相关系数

    # concat

    

    

    # fc

    # fc_1

    fc_1 = fc_layer(flatten, 512)

    # fc_2

    fc_2 = fc_layer(fc_1, 512)

                        

    # x_output

    x_output = Dense(units=11, activation="softmax")(fc_2)

    

    # 建立模型

    model = Model(inputs=raw_input, outputs=x_output)

    # 编译模型

    model.compile(Adam(3e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    

    return model
model = model_build()
generator = train_generator(X_train, Y_train, 256)
model.fit_generator(generator, steps_per_epoch=10000, epochs = 6, verbose=1)
ss = pd.read_csv('sample_submission.csv')
ss.head()
from tqdm import tqdm

X_test = test.signal.values

test_length = X_test.shape[0]

new_X_test = np.empty((test_length-100, 0, 1))

for i in tqdm(range(100)):

    new_X_test = np.append(new_X_test, X_test[i:test_length+i-100].reshape(-1,1, 1), axis=1)

    
Y_predict = model.predict(new_X_test, batch_size=64, verbose=1)
predict = np.argmax(Y_predict, axis=1)
predict = np.append(np.array([0 for i in range(80)]), predict)

predict = np.append(predict, np.array([0 for i in range(20)]))
ss.open_channels = predict
ss.to_csv('submission.csv', index=False, float_format='%.4f')
!kaggle competitions submit -c liverpool-ion-switching -f submission.csv -m "Message"