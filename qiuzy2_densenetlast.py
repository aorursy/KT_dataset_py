import os

import numpy as np

import pandas as pd

from skimage.io import imread

import matplotlib.pyplot as plt

import gc; gc.enable() 

print(os.listdir("../input/airbus-ship-detection"))
masks = pd.read_csv(os.path.join('../input/airbus-ship-detection', 'train_ship_segmentations_v2.csv'))

not_empty = pd.notna(masks.EncodedPixels)

print(not_empty.sum(), 'masks in', masks[not_empty].ImageId.nunique(), 'images')#非空图片中的mask数量

print((~not_empty).sum(), 'empty images in', masks.ImageId.nunique(), 'total images')#所有图片中非空图片

masks.head()
masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)

masks.head()
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()

unique_img_ids.head()
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)



unique_img_ids.head()
ship_dir = '../input/airbus-ship-detection'

train_image_dir = os.path.join(ship_dir, 'train_v2')

test_image_dir = os.path.join(ship_dir, 'test_v2')

unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])

unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id: 

                                                               os.stat(os.path.join(train_image_dir, 

                                                                                    c_img_id)).st_size/1024)

unique_img_ids.head()
unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb'] > 50] # keep only +50kb files

plt.hist(x = unique_img_ids['file_size_kb'], # 指定绘图数据

           bins = 6, # 指定直方图中条块的个数

           color = 'steelblue', # 指定直方图的填充色

           edgecolor = 'black' # 指定直方图的边框色

          )

plt.xticks([50,100,150,200,250,300,350,400,450,500])

plt.ylabel("number")

plt.xlabel('file_size_kb')

#unique_img_ids['file_size_kb'].hist()#绘制直方图

masks.drop(['ships'], axis=1, inplace=True)

unique_img_ids.sample(7)

plt.title("Number of images of each size")
print(unique_img_ids['ships'].value_counts())
train_0 = unique_img_ids[unique_img_ids['ships']==1].sample(1800)

train_1 = unique_img_ids[unique_img_ids['ships']==2].sample(1800)

train_2 = unique_img_ids[unique_img_ids['ships']==3].sample(1800)
train_3 = unique_img_ids[unique_img_ids['ships']!=3]

train_3 = train_3[unique_img_ids['ships']!=2]

train_3 = train_3[unique_img_ids['ships']!=1]
unique_img_ids=pd.concat([train_0,train_1,train_2,train_3])
SAMPLES_PER_GROUP = 10000

balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)

#图片有相同船舶数量，但超出2000的不要

rect=plt.hist(x = balanced_train_df['ships'], # 指定绘图数据

           bins = 16, # 指定直方图中条块的个数

           color = 'steelblue', # 指定直方图的填充色

           edgecolor = 'black' # 指定直方图的边框色

          )

plt.yticks(range(0,1800,300))

plt.xticks(range(0,15))

plt.ylabel("Number of images")

plt.xlabel('Number of ships')

plt.title("Number of images containing different number of vessels")

#balanced_train_df['ships'].hist(bins=balanced_train_df['ships'].max()+1)

print(balanced_train_df.shape[0], 'images',balanced_train_df.shape)#取出1万张图片

balanced_train_df=balanced_train_df.reset_index(drop = True)#删除原来的索引。

balanced_train_df=balanced_train_df.sample(frac=1.0)
from PIL import Image

x = np.empty(shape=(20188, 256,256,3),dtype=np.uint8)

y = np.empty(shape=20188,dtype=np.uint8)

for index, image in enumerate(balanced_train_df['ImageId']):

    image_array= Image.open('../input/airbus-ship-detection/train_v2/' + image).resize((256,256)).convert('RGB')

    x[index] = image_array

    y[index]=balanced_train_df[balanced_train_df['ImageId']==image]['has_ship'].iloc[0]



print(x.shape)

print(y.shape)
#Set target to one hot target for classification problem

#为分类问题将目标设置为一个热目标

from sklearn.preprocessing import OneHotEncoder

y_targets =y.reshape(len(y),-1)

enc = OneHotEncoder()

enc.fit(y_targets)

y = enc.transform(y_targets).toarray()

print(y.shape)
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val  = train_test_split(x,y,test_size = 0.2,random_state=1,stratify=y)

x_train.shape, x_val.shape, y_train.shape, y_val.shape


from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

 

import os

 

from keras import backend as K

from keras.models import Model

from keras.layers import Activation

from keras.layers import AveragePooling2D

from keras.layers import BatchNormalization

from keras.layers import Concatenate

from keras.layers import Conv2D

from keras.layers import Dense

from keras.layers import GlobalAveragePooling2D

from keras.layers import GlobalMaxPooling2D

from keras.layers import Input

from keras.layers import MaxPooling2D

from keras.layers import ZeroPadding2D

from keras.utils.data_utils import get_file

from keras.engine.topology import get_source_inputs

from keras.applications import imagenet_utils

from keras.applications.imagenet_utils import decode_predictions

from keras_applications.imagenet_utils import _obtain_input_shape 

 



 

def dense_block(x, blocks, name):

    """A dense block.

    密集的模块

    # Arguments

    参数

        x: input tensor.

        x: 输入参数

        blocks: integer, the number of building blocks.

        blocks: 整型，生成块的个数。

        name: string, block label.

        name: 字符串，块的标签

    # Returns

    返回

        output tensor for the block.

        为块输出张量

    """

    for i in range(blocks):

        x = conv_block(x, 32, name=name + '_block' + str(i + 1))

    return x

 

 

def transition_block(x, reduction, name):

    """A transition block.

    转换块

    # Arguments

    参数

        x: input tensor.

        x: 输入参数

        reduction: float, compression rate at transition layers.

        reduction: 浮点数，转换层的压缩率

        name: string, block label.

        name: 字符串，块标签

    # Returns

    返回

        output tensor for the block.

        块输出张量

    """

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,

                           name=name + '_bn')(x)

    x = Activation('relu', name=name + '_relu')(x)

    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,

               name=name + '_conv')(x)

    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)

    return x

 

 

def conv_block(x, growth_rate, name):

    """A building block for a dense block.

    密集块正在建立的块

    # Arguments

    参数

        x: input tensor.

        x: 输入张量

        growth_rate: float, growth rate at dense layers.

        growth_rate:浮点数，密集层的增长率。

        name: string, block label.

        name: 字符串，块标签

    # Returns

    返回

        output tensor for the block.

        块输出张量

    """

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,

                            name=name + '_0_bn')(x)

    x1 = Activation('relu', name=name + '_0_relu')(x1)

    x1 = Conv2D(4 * growth_rate, 1, use_bias=False,

                name=name + '_1_conv')(x1)

    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,

                            name=name + '_1_bn')(x1)

    x1 = Activation('relu', name=name + '_1_relu')(x1)

    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False,

                name=name + '_2_conv')(x1)

    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])

    return x

 

 

def DenseNet(blocks,

             include_top=True,

             weights='imagenet',

             input_tensor=None,

             input_shape=None,

             pooling=None,

             classes=2):

    """Instantiates the DenseNet architecture.

    实例化DenseNet结构

    Optionally loads weights pre-trained

    on ImageNet. Note that when using TensorFlow,

    for best performance you should set

    `image_data_format='channels_last'` in your Keras config

    at ~/.keras/keras.json.

    可选择加载预训练的ImageNet权重。注意，如果是Tensorflow，最好在Keras配置中设置`image_data_format='channels_last'

    The model and the weights are compatible with

    TensorFlow, Theano, and CNTK. The data format

    convention used by the model is the one

    specified in your Keras config file.

    模型和权重兼容TensorFlow, Theano, and CNTK.模型使用的数据格式约定是Keras配置文件中指定的一种格式。

    # Arguments

    参数

        blocks: numbers of building blocks for the four dense layers.

        blocks: （构建）4个密集层需要块数量

        include_top: whether to include the fully-connected

            layer at the top of the network.

        include_top: 在网络的顶层（一般指最后一层）师傅包含全连接层

        weights: one of `None` (random initialization),

              'imagenet' (pre-training on ImageNet),

              or the path to the weights file to be loaded.

              以下的一个：`None` (随机初始化),'imagenet' (ImageNet预训练),或者下载权重文件的路径。

        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)

            to use as image input for the model.

        input_tensor: 可选的Keras张量（即，`layers.Input()`的输出），用作模型的图像输入。

        input_shape: optional shape tuple, only to be specified

            if `include_top` is False (otherwise the input shape

            has to be `(224, 224, 3)` (with `channels_last` data format)

            or `(3, 224, 224)` (with `channels_first` data format).

            It should have exactly 3 inputs channels.

        input_shape: 可选的形状元组，只有`include_top`是False（否则，输入形状必须

        是“（224, 224, 3）”（带有`channels_first` 数据格式。））时需要确认，它应该有3个输入通道。

        pooling: optional pooling mode for feature extraction

            when `include_top` is `False`.

            可选，当 `include_top`是FALSE，特征提取的池化模式。

            - `None` means that the output of the model will be

                the 4D tensor output of the

                last convolutional layer.

                `None` 表示，模型输出层是4维张量，从上一个的卷积层输出。

            - `avg` means that global average pooling

                will be applied to the output of the

                last convolutional layer, and thus

                the output of the model will be a 2D tensor.

              `avg`表示全局平均池化被应用到上一个的卷积层输出，所以模型输出是2维张量。

            - `max` means that global max pooling will

                be applied.

              `max`  表示全局最大池化被应用

        classes: optional number of classes to classify images

            into, only to be specified if `include_top` is True, and

            if no `weights` argument is specified.

        classes: 可选的类数分类的图像，只有指定，如果'include_top'是真的，如果没有'weights'参数被指定。

    # Returns

    返回

        A Keras model instance.

        一个Keras模型实例

    # Raises

    补充

        ValueError: in case of invalid argument for `weights`,

            or invalid input shape.

        ValueError: weights`无效的参数，或者无效的输入形状

    """

    if not (weights in {'imagenet', None} or os.path.exists(weights)):

        raise ValueError('The `weights` argument should be either '

                         '`None` (random initialization), `imagenet` '

                         '(pre-training on ImageNet), '

                         'or the path to the weights file to be loaded.')

 

    if weights == 'imagenet' and include_top and classes != 1000:

        raise ValueError('If using `weights` as imagenet with `include_top`'

                         ' as true, `classes` should be 1000')

 

    # Determine proper input shape

    input_shape = _obtain_input_shape(input_shape,

                                      default_size=256,

                                      min_size=221,

                                      data_format=K.image_data_format(),

                                      require_flatten=include_top,

                                      weights=weights)

 

    if input_tensor is None:

        img_input = Input(shape=input_shape)

    else:

        if not K.is_keras_tensor(input_tensor):

            img_input = Input(tensor=input_tensor, shape=input_shape)

        else:

            img_input = input_tensor

 

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

 

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)

    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,

                           name='conv1/bn')(x)

    x = Activation('relu', name='conv1/relu')(x)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)

    x = MaxPooling2D(3, strides=2, name='pool1')(x)

 

    x = dense_block(x, blocks[0], name='conv2')

    x = transition_block(x, 0.5, name='pool2')

    x = dense_block(x, blocks[1], name='conv3')

    x = transition_block(x, 0.5, name='pool3')

    x = dense_block(x, blocks[2], name='conv4')

    x = transition_block(x, 0.5, name='pool4')

    x = dense_block(x, blocks[3], name='conv5')

 

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,

                           name='bn')(x)

 

    if include_top:

        x = GlobalAveragePooling2D(name='avg_pool')(x)

        x = Dense(classes, activation='sigmoid', name='fc1000')(x)

    else:

        if pooling == 'avg':

            x = GlobalAveragePooling2D(name='avg_pool')(x)

        elif pooling == 'max':

            x = GlobalMaxPooling2D(name='max_pool')(x)

 

    # Ensure that the model takes into account

    # any potential predecessors of `input_tensor`.

    # 确保模型考虑到任何潜在的前缀“input_tensor”。

    if input_tensor is not None:

        inputs = get_source_inputs(input_tensor)

    else:

        inputs = img_input

 

    # Create model.

    # 建立模型

    if blocks == [6, 12, 24, 16]:

        model = Model(inputs, x, name='densenet121')

    elif blocks == [6, 12, 32, 32]:

        model = Model(inputs, x, name='densenet169')

    elif blocks == [6, 12, 48, 32]:

        model = Model(inputs, x, name='densenet201')

    else:

        model = Model(inputs, x, name='densenet')

 

 

    return model

 

 

def DenseNet121(include_top=True,

                weights=None,

                input_tensor=None,

                input_shape=None,

                pooling=None,

                classes=2):

    return DenseNet([6, 12, 24, 16],

                    include_top, weights,

                    input_tensor, input_shape,

                    pooling, classes)

model_final =DenseNet121()
from keras.callbacks import Callback

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(Callback):

    def on_train_begin(self, logs={}):

        self.val_f1s = []

        self.val_recalls = []

        self.val_precisions = []



    def on_epoch_end(self, epoch, logs={}):

#         val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()

        val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)

#         val_targ = self.validation_data[1]

        val_targ = np.argmax(self.validation_data[1], axis=1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')

        _val_recall = recall_score(val_targ, val_predict)

        _val_precision = precision_score(val_targ, val_predict)

        self.val_f1s.append(_val_f1)

        self.val_recalls.append(_val_recall)

        self.val_precisions.append(_val_precision)

        print('— val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))

#         print(' — val_f1:' ,_val_f1)

        return



metrics1 = Metrics()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weight_path="{}_weights.best.hdf5".format('boat_detector')



checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)



reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=3) # probably needs to be more patient, but kaggle time is limited

callbacks_list = [checkpoint, early, reduceLROnPlat,metrics1]
from keras import optimizers

def fit():

    epochs = 40

    lrate = 0.01

    decay = lrate/epochs

    #adam = optimizers.Adam(lr=lrate,beta_1=0.9, beta_2=0.999, decay=decay)

    sgd = optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

    model_final.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['binary_accuracy'])

    loss_history=[model_final.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=40, batch_size=50,callbacks=callbacks_list)]

    

    return loss_history

num=0



while True:

    num=num+1

#     prefix='%d'%(num)

    loss_history = fit()

    model_final.save_weights('my_model_weights%d.h5'% num)

    if np.min([mh.history['val_loss'] for mh in loss_history]) < 0.1:

        break

    if num==1:

        break
def show_loss(loss_history):

    epochs = np.concatenate([mh.epoch for mh in loss_history])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    

    _ = ax1.plot(epochs, np.concatenate([mh.history['loss'] for mh in loss_history]), 'b-',

                 epochs, np.concatenate([mh.history['val_loss'] for mh in loss_history]), 'r-')

    ax1.legend(['Training', 'Validation'])#图表，损失函数（训练和验证）的迭代图表

    ax1.set_title('Loss')

    

    _ = ax2.plot(epochs, np.concatenate([mh.history['binary_accuracy'] for mh in loss_history]), 'b-',

                 epochs, np.concatenate([mh.history['val_binary_accuracy'] for mh in loss_history]), 'r-')

    ax2.legend(['Training', 'Validation'])#准确率，（训练和迭代的）

    ax2.set_title('Binary Accuracy (%)')



show_loss(loss_history)
unique_img_ids1 = unique_img_ids[20000:30000]
x_test = np.empty(shape=(10000, 256,256,3),dtype=np.uint8)#10680 256

y_test = np.empty(shape=10000,dtype=np.uint8)

for index, image in enumerate(unique_img_ids1['ImageId']):

    image_array= Image.open('../input/airbus-ship-detection/train_v2/' + image).resize((256,256)).convert('RGB') #256

    x_test[index] = image_array

    y_test[index]=unique_img_ids1[unique_img_ids1['ImageId']==image]['has_ship'].iloc[0]



print(x_test.shape)

print(y_test.shape)
y_test_targets =y_test.reshape(len(y_test),-1)

enc = OneHotEncoder()

enc.fit(y_test_targets)

y_test = enc.transform(y_test_targets).toarray()

print(y_test.shape)
predict_ship = model_final.evaluate( x_test,y_test)

acc=predict_ship[1]*100
print ('Accuracy of random data = '+ str(acc) + "%")