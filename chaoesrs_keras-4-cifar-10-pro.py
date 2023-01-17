import pandas as pd

import numpy as np

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import keras

keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,  #布尔值。将输入数据的均值设置为 0，逐特征进行

                                             samplewise_center=False,   #布尔值。将每个样本的均值设置为 0

                                             featurewise_std_normalization=False, #布尔值。将输入除以数据标准差，逐特征进行

                                             samplewise_std_normalization=False, #布尔值。将每个输入除以其标准差

                                             zca_whitening=False,       #布尔值。是否应用 ZCA 白化 

                                             zca_epsilon=1e-06,         #ZCA 白化的 epsilon 值，默认为 1e-6

                                             rotation_range=0,          #整数。随机旋转的度数范围

                                            width_shift_range=0.0,     #浮点数、一维数组或整数 如果<1，则是除以总宽度的值，或者如果 >=1，则为像素值

                                             height_shift_range=0.0,    #浮点数、一维数组或整数 如果 <1，则是除以总宽度的值，或者如果 >=1，则为像素值

                                                                         #两者共同确定剪切的尺寸

                                             brightness_range=None,     #随机亮度范围

                                             shear_range=0.0,           #浮点数。剪切强度（以弧度逆时针方向剪切角度）

                                             zoom_range=0.0,            #浮点数 或 [lower, upper]。随机缩放范围。

                                                                        #如果是浮点数，[lower, upper] = [1-zoom_range, 1+zoom_range]

                                             channel_shift_range=0.0,   #浮点数。随机通道转换的范围

                                             fill_mode='nearest',       #{"constant", "nearest", "reflect" or "wrap"} 之一。

                                                                        #默认为 'nearest'。输入边界以外的点根据给定的模式填充：

                                                                        #'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)

                                                                        #'nearest': aaaaaaaa|abcd|dddddddd

                                                                        #'reflect': abcddcba|abcd|dcbaabcd

                                                                        #'wrap': abcdabcd|abcd|abcdabcd

                                             cval=0.0,                  #浮点数或整数。用于边界之外的点的值，当 fill_mode = "constant" 时

                                             horizontal_flip=False,     #布尔值。随机水平翻转

                                             vertical_flip=False,       #布尔值。随机垂直翻转

                                             rescale=None,              #重缩放因子。默认为 None。如果是 None 或 0，不进行缩放，否则将数据乘以所提供的值（在应用任何其他转换之前）

                                             preprocessing_function=None, # 应用于每个输入的函数。这个函数会在任何其他改变之前运行。

                                                                        #这个函数需要一个参数：一张图像（秩为 3 的 Numpy 张量），并且应该输出一个同尺寸的 Numpy 张量。

                                             data_format=None,          #图像数据格式，{"channels_first", "channels_last"} 之一。

                                                                        #"channels_last" 模式表示图像输入尺寸应该为 (samples, height, width, channels)

                                                                        #"channels_first" 模式表示输入尺寸应该为 (samples, channels, height, width)。

                                                                        #默认为 在 Keras 配置文件 ~/.keras/keras.json 中的 image_data_format 值。

                                                                        #如果你从未设置它，那它就是 "channels_last"

                                             validation_split=0.0,      #浮点数。保留用于验证的图像的比例（严格在0和1之间）

                                             dtype=None)                #生成数组使用的数据类型
import pickle,keras

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',

               'dog', 'frog', 'horse', 'ship', 'truck']

data_dir='/kaggle/input/cifar10-python/cifar-10-batches-py/'

images_train=[]

labels_train=[]

def Load_Data():

    for i in range (5):

        filepath=os.path.join(data_dir,'data_batch_%d' %(i+1))

        print('loading data',filepath)

        with open (filepath,'rb') as f:

            data_dict=pickle.load(f, encoding='latin1')

            images_batch = data_dict['data']

            labels_batch = data_dict['labels']

            images_batch = images_batch.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")

            labels_batch = np.array(labels_batch)

        images_train.append(images_batch)

        labels_train.append(labels_batch)

        X_train=np.concatenate(images_train)

        Y_train=np.concatenate(labels_train)

    filepath=os.path.join(data_dir,'test_batch')

    images_test=[]

    labels_test=[]

    with open (filepath,'rb') as f:

        print('loading data',filepath)

        data_dict=pickle.load(f, encoding='latin1')

        images_batch = data_dict['data']

        labels_batch = data_dict['labels']

        images_batch = images_batch.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")

        labels_batch = np.array(labels_batch)

        images_test.append(images_batch)

        labels_test.append(labels_batch)

        X_test=np.concatenate(images_test)

        Y_test=np.concatenate(labels_test)

        print('finished loading ')

        return X_train,Y_train,X_test,Y_test

X_train,Y_train,X_test,Y_test=Load_Data()

print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
X_train_normalize=X_train.astype('float32')/255.0

X_test_normalize=X_test.astype('float32')/255.0
from keras.utils import np_utils

Y_train_onehot=np_utils.to_categorical(Y_train)

Y_test_onehot=np_utils.to_categorical(Y_test)
change=keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,vertical_flip=True)

change.fit(X_test_normalize)
import keras

from keras import applications

from keras.layers import Dense,Flatten,Dropout

from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions

VGGmodel=VGG16(weights='imagenet',include_top=False,input_shape=(32,32,3))

for layers in VGGmodel.layers:

    layers.trainable = False#「冻结」一个层意味着将其排除在训练之外，即其权重将永远不会更新。这在微调模型或使用固定的词向量进行文本输入中很有用。

#加入全连接层进行微调

model = Flatten()(VGGmodel.output)

model = Dense(4096, activation='relu')(model)

model = Dropout(0.5)(model)

model = Dense(4096, activation='relu')(model)

model = Dropout(0.5)(model)

model = Dense(10, activation='softmax')(model)

VGGnet= keras.Model(inputs=VGGmodel.input, outputs=model)

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

VGGnet.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

VGGnet.fit(change.flow(X_train_normalize, Y_train_onehot, batch_size=32),epochs=10)
print(VGGnet.evaluate(X_test_normalize, Y_test_onehot, verbose=0))