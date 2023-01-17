import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
import time
#print(os.listdir("../input"))
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.layers import Dropout
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
TRAIN_DIR = '../input/dogs-vs-cats-redux-kernels-edition/train/'
TEST_DIR = '../input/dogs-vs-cats-redux-kernels-edition/test/'
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]
test_id = np.array([int(i[i.find('test/')+5:-4]) for i in test_images],ndmin=2).T
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) # read img into color mode
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = img-np.array([123.68, 116.779, 103.939]).reshape((1,1,3))
    return img 
def prep_data(images):
    labels = []
    count = len(images)
    data = np.ndarray((count, 224, 224,3), dtype=np.float32)
    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
        if image_file[image_file.find('train/')+6:image_file.find('train/')+9] =='dog':
            labels.append(1)
        else:
            labels.append(0)
    return data, labels
trainset = train_images[15000:17000]
validationset = train_images[17000:17800]
train, train_labels = prep_data(trainset)
validation, val_labels= prep_data(validationset)
def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x
def ResNet50(include_top=True, weights=None,input_tensor=None, input_shape=None,pooling=None,classes=1000):
    input_shape = _obtain_input_shape(input_shape,default_size=224,min_size=197,data_format=K.image_data_format(),
                                      require_flatten=include_top)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes,kernel_initializer='lecun_normal' ,activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        else:
            weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')
    return model

image_input = Input(shape=(224, 224, 3))
model = ResNet50(input_tensor=image_input, include_top=False,weights='imagenet')
#model.summary()
last_layer = model.output
x = GlobalAveragePooling2D()(last_layer)
x = Dropout(0.1)(x)
x = Dense(256, kernel_initializer='lecun_normal', activation='relu',name='fc-2')(x)
x = Dropout(0.3)(x)
out = Dense(1, activation='sigmoid',name='output_layer')(x)
custom_resnet_model = Model(inputs=model.input, outputs=out)
for layer in custom_resnet_model.layers[:-5]:
    layer.trainable = False
custom_resnet_model.layers[-5].trainable
#custom_resnet_model.summary()
datagen = image.ImageDataGenerator()
train_generator = datagen.flow(train,train_labels,batch_size=32)
val_generator = datagen.flow(validation,val_labels,batch_size=16)
custom_resnet_model.compile(loss='binary_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
custom_resnet_model.fit_generator(train_generator, steps_per_epoch=50,validation_data=val_generator, validation_steps=20,epochs=5, verbose=1)
custom_resnet_model.save('dog_cat_round1.h')
custom_resnet_model.fit_generator(train_generator, steps_per_epoch=50,validation_data=val_generator, validation_steps=20,epochs=5, verbose=1)
custom_resnet_model.save('dog_cat_round2.h')