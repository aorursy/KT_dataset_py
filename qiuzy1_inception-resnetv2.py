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
import numpy as np



from keras.preprocessing import image

from keras.models import Model

from keras.layers import Activation,AveragePooling2D,BatchNormalization,Concatenate

from keras.layers import Conv2D,Dense,GlobalAveragePooling2D,GlobalMaxPooling2D,Input,Lambda,MaxPooling2D

from keras.applications.imagenet_utils import decode_predictions

from keras.utils.data_utils import get_file

from keras import backend as K

BASE_WEIGHT_URL = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.7/'



def conv2d_bn(x,filters,kernel_size,strides=1,padding='same',activation='relu',use_bias=False,name=None):

    x = Conv2D(filters,

               kernel_size,

               strides=strides,

               padding=padding,

               use_bias=use_bias,

               name=name)(x)

    if not use_bias:

        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3

        bn_name = None if name is None else name + '_bn'

        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    if activation is not None:

        ac_name = None if name is None else name + '_ac'

        x = Activation(activation, name=ac_name)(x)

    return x



def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):

    if block_type == 'block35':

        branch_0 = conv2d_bn(x, 32, 1)

        branch_1 = conv2d_bn(x, 32, 1)

        branch_1 = conv2d_bn(branch_1, 32, 3)

        branch_2 = conv2d_bn(x, 32, 1)

        branch_2 = conv2d_bn(branch_2, 48, 3)

        branch_2 = conv2d_bn(branch_2, 64, 3)

        branches = [branch_0, branch_1, branch_2]

    elif block_type == 'block17':

        branch_0 = conv2d_bn(x, 192, 1)

        branch_1 = conv2d_bn(x, 128, 1)

        branch_1 = conv2d_bn(branch_1, 160, [1, 7])

        branch_1 = conv2d_bn(branch_1, 192, [7, 1])

        branches = [branch_0, branch_1]

    elif block_type == 'block8':

        branch_0 = conv2d_bn(x, 192, 1)

        branch_1 = conv2d_bn(x, 192, 1)

        branch_1 = conv2d_bn(branch_1, 224, [1, 3])

        branch_1 = conv2d_bn(branch_1, 256, [3, 1])

        branches = [branch_0, branch_1]

    else:

        raise ValueError('Unknown Inception-ResNet block type. '

                         'Expects "block35", "block17" or "block8", '

                         'but got: ' + str(block_type))



    block_name = block_type + '_' + str(block_idx)

    mixed = Concatenate(name=block_name + '_mixed')(branches)

    up = conv2d_bn(mixed,K.int_shape(x)[3],1,activation=None,use_bias=True,name=block_name + '_conv')



    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,

               output_shape=K.int_shape(x)[1:],

               arguments={'scale': scale},

               name=block_name)([x, up])

    if activation is not None:

        x = Activation(activation, name=block_name + '_ac')(x)

    return x





def InceptionResNetV2(input_shape=[256,256,3],

                      classes=2):



    input_shape = [256,256,3]



    img_input = Input(shape=input_shape)



    # Stem block: 299,299,3 -> 35 x 35 x 192

    x = conv2d_bn(img_input, 32, 3, strides=2, padding='valid')

    x = conv2d_bn(x, 32, 3, padding='valid')

    x = conv2d_bn(x, 64, 3)

    x = MaxPooling2D(3, strides=2)(x)

    x = conv2d_bn(x, 80, 1, padding='valid')

    x = conv2d_bn(x, 192, 3, padding='valid')

    x = MaxPooling2D(3, strides=2)(x)



    # Mixed 5b (Inception-A block):35 x 35 x 192 -> 35 x 35 x 320

    branch_0 = conv2d_bn(x, 96, 1)

    branch_1 = conv2d_bn(x, 48, 1)

    branch_1 = conv2d_bn(branch_1, 64, 5)

    branch_2 = conv2d_bn(x, 64, 1)

    branch_2 = conv2d_bn(branch_2, 96, 3)

    branch_2 = conv2d_bn(branch_2, 96, 3)

    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 64, 1)

    branches = [branch_0, branch_1, branch_2, branch_pool]



    x = Concatenate(name='mixed_5b')(branches)



    # 10次Inception-ResNet-A block:35 x 35 x 320 -> 35 x 35 x 320

    for block_idx in range(1, 11):

        x = inception_resnet_block(x,

                                   scale=0.17,

                                   block_type='block35',

                                   block_idx=block_idx)



    # Reduction-A block:35 x 35 x 320 -> 17 x 17 x 1088

    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid')

    branch_1 = conv2d_bn(x, 256, 1)

    branch_1 = conv2d_bn(branch_1, 256, 3)

    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')

    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)

    branches = [branch_0, branch_1, branch_pool]

    x = Concatenate(name='mixed_6a')(branches)



    # 20次Inception-ResNet-B block: 17 x 17 x 1088 -> 17 x 17 x 1088 

    for block_idx in range(1, 21):

        x = inception_resnet_block(x,

                                   scale=0.1,

                                   block_type='block17',

                                   block_idx=block_idx)





    # Reduction-B block: 17 x 17 x 1088 -> 8 x 8 x 2080

    branch_0 = conv2d_bn(x, 256, 1)

    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')

    branch_1 = conv2d_bn(x, 256, 1)

    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')

    branch_2 = conv2d_bn(x, 256, 1)

    branch_2 = conv2d_bn(branch_2, 288, 3)

    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')

    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)

    branches = [branch_0, branch_1, branch_2, branch_pool]

    x = Concatenate(name='mixed_7a')(branches)



    # 10次Inception-ResNet-C block: 8 x 8 x 2080 -> 8 x 8 x 2080

    for block_idx in range(1, 10):

        x = inception_resnet_block(x,

                                   scale=0.2,

                                   block_type='block8',

                                   block_idx=block_idx)

    x = inception_resnet_block(x,

                               scale=1.,

                               activation=None,

                               block_type='block8',

                               block_idx=10)



    # 8 x 8 x 2080 -> 8 x 8 x 1536

    x = conv2d_bn(x, 1536, 1, name='conv_7b')



    x = GlobalAveragePooling2D(name='avg_pool')(x)

    x = Dense(classes, activation='sigmoid', name='predictions')(x)



    inputs = img_input



    # 创建模型

    model = Model(inputs, x, name='inception_resnet_v2')



    return model

# from keras.applications.vgg16 import VGG16 as PTModel



#from keras.applications.resnet50 import ResNet50 as PTModel



#from keras.applications.inception_v3 import InceptionV3 as PTModel



#from keras.applications.xception import Xception as PTModel



#from keras.applications.densenet import DenseNet169 as PTModel



#from keras.applications.densenet import DenseNet121 as PTModel



# from keras.applications.resnet50 import ResNet50 as PTModel
# img_width, img_height = 256, 256

# model = PTModel(weights = None, include_top=False, input_shape = (img_width, img_height, 3))

# #weights=None，‘imagenet’表示不加载权重
# from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

# from keras.models import Sequential, Model 

# from keras import backend as K

# for layer in model.layers:

#     layer.trainable = False



# x = model.output

# x = Flatten()(x)

# x = Dense(1024, activation="relu")(x)

# x = Dropout(0.5)(x)

# x = Dense(1024, activation="relu")(x)

# predictions = Dense(2, activation="sigmoid")(x)



# # creating the final model创建最终模型

# model_final = Model(input = model.input, output = predictions)
model_final = InceptionResNetV2()
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

    epochs = 20

    lrate = 0.01

    decay = lrate/epochs

    #adam = optimizers.Adam(lr=lrate,beta_1=0.9, beta_2=0.999, decay=decay)

    sgd = optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

    model_final.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['binary_accuracy'])

    loss_history=[model_final.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=20, batch_size=50,callbacks=callbacks_list)]

    

    return loss_history

num=0



while True:

    num=num+1

#     prefix='%d'%(num)

    loss_history = fit()

    model_final.save_weights('my_model_weights%d.h5'% num)

    if np.min([mh.history['val_loss'] for mh in loss_history]) < 0.01:

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