import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras import layers
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#defaults: 
# To use additional dataset. 
USE60KDATASET = True 
train_df_1 = pd.read_csv('../input/digit-recognizer/train.csv')
test_df = pd.read_csv('../input/digit-recognizer/test.csv')
train_df_2 = pd.read_csv('../input/mnist-in-csv/mnist_train.csv')


train_df_1.head(3)
train_df_2.head(3)

train_df_2.columns = train_df_1.columns
if USE60KDATASET:
    train_df = pd.concat([train_df_1,train_df_2])
else:
    train_df = train_df_1 

train_df.shape
train_df.isnull().any().describe()
test_df.isnull().any().describe()
leftjoin = pd.merge(train_df, test_df, how='left',indicator=True)
train_df = leftjoin[leftjoin['_merge'] == 'left_only'].drop(columns=['_merge'])

train_df.shape
train_label = train_df.iloc[:,0].reset_index().iloc[:,1:]
train_in = train_df.iloc[:,1:].reset_index().iloc[:,1:]

train_in = train_in/255
test_in = test_df/255
train_in = np.asarray([x.reshape(28,28,1)for x in train_in.values])
test_in = np.asarray([x.reshape(28,28,1)for x in test_in.values])
labels_flat = train_label.label.ravel()

labels_one_hot_test = np.zeros((labels_flat.shape[0], 10))
hh = (np.arange(labels_flat.shape[0])*10 + labels_flat.ravel()).astype(int)

labels_one_hot_test.flat[hh] = 1
if K.image_data_format() == 'channels_first':
    channel_axis = 1
else:
    channel_axis = 3
def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x


input_tensor = layers.Input(shape=(28,28,1))

x = conv2d_bn(input_tensor, 4, 3, 3, strides=(1, 1), padding='valid')
x = conv2d_bn(x, 4, 3, 3, padding='valid')
x = conv2d_bn(x, 8, 3, 3, padding='valid')
x = layers.MaxPooling2D((3, 3), strides=(1, 1))(x)

x = conv2d_bn(x, 12, 1, 1, padding='valid')
x = conv2d_bn(x, 32, 3, 3, padding='valid')
x = layers.MaxPooling2D((3, 3), strides=(1, 1))(x)


branch1x1 = conv2d_bn(x, 8, 1, 1,padding='valid')   # 1,1

branch5x5 = conv2d_bn(x, 6, 1, 1,padding='valid') # 1,1
branch5x5 = conv2d_bn(branch5x5, 8, 5, 5) # 5,5 

branch3x3dbl = conv2d_bn(x, 8, 1, 1,padding='valid') # 1,1
branch3x3dbl = conv2d_bn(branch3x3dbl, 12, 3, 3)
branch3x3dbl = conv2d_bn(branch3x3dbl, 12, 3, 3)

branch_pool = layers.AveragePooling2D((3, 3),  strides=(1, 1),  padding='same')(x)
branch_pool = conv2d_bn(branch_pool, 4, 1, 1)
x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis,name='mixed0')


branch3x3 = conv2d_bn(x, 32, 3, 3, strides=(2, 2), padding='valid')

branch3x3dbl = conv2d_bn(x, 8, 1, 1)
branch3x3dbl = conv2d_bn(branch3x3dbl, 12, 3, 3)
branch3x3dbl = conv2d_bn(
    branch3x3dbl, 12, 3, 3, strides=(2, 2), padding='valid')

branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
x = layers.concatenate(
    [branch3x3, branch3x3dbl, branch_pool],
    axis=channel_axis,
    name='mixed1')


branch1x1 = conv2d_bn(x, 32, 1, 1,padding='valid') # 1 1

branch7x7 = conv2d_bn(x, 16, 1, 1,padding='valid') # 1 1
branch7x7 = conv2d_bn(branch7x7, 16, 1, 7)
branch7x7 = conv2d_bn(branch7x7, 32, 7, 1)

branch7x7dbl = conv2d_bn(x, 16, 1, 1,padding='valid') # 1 1
branch7x7dbl = conv2d_bn(branch7x7dbl, 16, 7, 1)
branch7x7dbl = conv2d_bn(branch7x7dbl, 16, 1, 7)
branch7x7dbl = conv2d_bn(branch7x7dbl, 16, 7, 1)
branch7x7dbl = conv2d_bn(branch7x7dbl, 32, 1, 7)

branch_pool = layers.AveragePooling2D((3, 3),
                                      strides=(1, 1 ),
                                      padding='same')(x)
branch_pool = conv2d_bn(branch_pool, 32, 1, 1,padding='valid') # 1 1
x = layers.concatenate(
    [branch1x1, branch7x7, branch7x7dbl, branch_pool],
    axis=channel_axis,
    name='mixed2')


branch3x3 = conv2d_bn(x, 32, 1, 1,padding='valid') # 1 1
branch3x3 = conv2d_bn(branch3x3, 64, 3, 3,
                      strides=(2, 2), padding='valid')

branch7x7x3 = conv2d_bn(x, 32, 1, 1,padding='valid') # 1 1
branch7x7x3 = conv2d_bn(branch7x7x3, 32, 1, 7)
branch7x7x3 = conv2d_bn(branch7x7x3, 32, 7, 1)
branch7x7x3 = conv2d_bn(
    branch7x7x3, 32, 3, 3, strides=(2, 2), padding='valid')

branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2),padding='valid')(x) # strides 2 2
x = layers.concatenate(
    [branch3x3, branch7x7x3, branch_pool],
    axis=channel_axis,
    name='mixed3')



branch1x1 = conv2d_bn(x, 32, 1, 1,padding='valid',name='validcheck') # 1 1

branch3x3 = conv2d_bn(x, 32, 1, 1,padding='valid')
branch3x3_1 = conv2d_bn(branch3x3, 32, 1, 3)
branch3x3_2 = conv2d_bn(branch3x3, 32, 3, 1)
branch3x3 = layers.concatenate(
    [branch3x3_1, branch3x3_2],
    axis=channel_axis,
    name='mixed4')

branch3x3dbl = conv2d_bn(x, 32, 1, 1,padding='valid')
branch3x3dbl = conv2d_bn(branch3x3dbl, 32, 3, 3)
branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 32, 1, 3)
branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 32, 3, 1)
branch3x3dbl = layers.concatenate(
    [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

branch_pool = layers.AveragePooling2D(
    (3, 3), strides=(1, 1), padding='same')(x)
branch_pool = conv2d_bn(branch_pool, 32, 1, 1,padding='valid') # 1 1 
x = layers.concatenate(
    [branch1x1, branch3x3, branch3x3dbl, branch_pool],
    axis=channel_axis,
    name='mixed5')
        
      
x = layers.AveragePooling2D(
    (3, 3), strides=(1, 1), padding='valid')(x)
x = layers.Dropout(0.01)(x)
x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
x = layers.Dense(64, activation = "relu")(x)
x = layers.Dense(10, activation='softmax', name='predictions')(x)
model = Model(input_tensor, x)
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
modelfit = model.fit(x=train_in, y=labels_one_hot_test, batch_size=100, epochs=5, verbose=1, callbacks=None, validation_split=0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.15, # Randomly zoom image 
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(train_in)
modelfitwithdatagen = model.fit_generator(datagen.flow(train_in,labels_one_hot_test, batch_size=100),
                              epochs = 5, validation_data = None,
                              verbose = 1, steps_per_epoch=779)

pred = model.predict(test_in)

pred_val = np.argmax(pred,axis=1)

results = pd.Series(pred_val,name="Label")
submission = pd.concat([pd.Series(range(1,pred_val.shape[0]+1),name = "ImageId"),results],axis = 1)

submission.to_csv("Results.csv",index=False)

