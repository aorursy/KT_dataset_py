# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# !pip install keras --upgrade

import keras
from keras import applications
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dropout, Flatten, Dense, Conv2D, LeakyReLU, BatchNormalization, Activation, AveragePooling2D, concatenate, GlobalAveragePooling2D, MaxPooling2D
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint

import glob

# need use keras 2.2.x
# assert keras.__version__ > '2.2.0'
# Any results you write to the current directory are saved as output.
test = pd.DataFrame()
test['file'] = glob.glob("../input/walk_or_run_test/test/run/*")+glob.glob("../input/walk_or_run_test/test/walk/*")
test['label'] = [ 1 for _ in glob.glob("../input/walk_or_run_test/test/run/*")]+[0 for _ in glob.glob("../input/walk_or_run_test/test/walk/*")]

train = pd.DataFrame()
train['file'] = glob.glob("../input/walk_or_run_train/train/run/*")+glob.glob("../input/walk_or_run_train/train/walk/*")
train['label'] = [ 1 for _ in glob.glob("../input/walk_or_run_train/train/run/*")]+[0 for _ in glob.glob("../input/walk_or_run_train/train/walk/*")]
train = train.sample(frac=1).reset_index(drop=True)
train.head()
def load_base(model):
    if model == "vgg" or model == "vgg16":
        return applications.VGG16(weights='imagenet', include_top=False),  applications.vgg16.decode_predictions, applications.vgg16.preprocess_input
    elif model == "mobilenet" or model == "mn":
        return applications.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3)), applications.mobilenet.decode_predictions, applications.mobilenet.preprocess_input
    elif model == "resnet" or model == "resnet50":
        resnet = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        if(keras.__version__ < '2.2.0'):    
            resnet = Model(resnet.input, resnet.layers[-2].output)
        return resnet, applications.resnet50.decode_predictions, applications.resnet50.preprocess_input
    elif model == "inceptionv3" or model == "inception":
        return applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3)), applications.inception_v3.decode_predictions, applications.inception_v3.preprocess_input
    elif model == "xception":
        return applications.Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3)), applications.xception.decode_predictions, applications.xception.preprocess_input
    elif model == "densenet":
        return applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3)), applications.densenet.decode_predictions, applications.densenet.preprocess_input
    elif model == "mn2":
        return applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)), applications.mobilenetv2.decode_predictions, applications.mobilenetv2.preprocess_input
    elif model == "ir2" or model == "inception_resnet_v2":
        return applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)), applications.inception_resnet_v2.decode_predictions, applications.inception_resnet_v2.preprocess_input
    
    else:
        return None
# build model

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    filters = int(filters)
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=name + "_conv")(x)
    x = BatchNormalization(scale=False, name=name + "_bn")(x)
    x = Activation('relu', name=name)(x)
    return x 

def incept(x, name="incept", scale=1):
    branch1x1 = conv2d_bn(x, 64 // scale, 1, 1, name = name + "-1x1")

    branch5x5 = conv2d_bn(x, 48 // scale , 1, 1, name = name + "-5x5-1x1")
    branch5x5 = conv2d_bn(branch5x5, 64 // scale, 5, 5, name = name + "-5x5-5x5")

    branch3x3dbl = conv2d_bn(x, 64 // scale, 1, 1, name = name + "-3x3-1x1")
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96 // scale, 3, 3, name = name + "-3x3-3x3-1")
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96 // scale, 3, 3, name = name + "-3x3-3x3-2")

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32 // scale, 1, 1, name = name + "-pool")
    return concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        name= name + '-all')
  
def build_head_model(input_shape, n_classes = 120):
    head_input = Input(shape=input_shape, name = 'head_input')
    x = head_input
    x = Dropout(0.8)(x)
    x = incept(x, name="i2", scale=2)
    x = Flatten()(x)
    x = Dropout(0.8)(x)
    x = Dense(n_classes, activation='softmax', name='prediction')(x)
    return Model(input = head_input, output = x, name="dogs_ft_head_model")
  
def build_model(pretrained_model_name, n_classes):
    input = Input(shape=(224,224,3), name = 'image_input')
    x = input
    pretrained_model, _, preprocess = load_base(pretrained_model_name)
    for layer in pretrained_model.layers[:-1]:
        layer.trainable = False
    
    x = pretrained_model(x)
    head_model = build_head_model(x.get_shape()[1:].as_list(), n_classes)
    x = head_model(x)
  
    model = Model(input = input, output = x)
    model.compile(metrics=["accuracy"],
                  loss="categorical_crossentropy",
                  optimizer=optimizers.Adadelta())
    
    return model, head_model, preprocess
images_root = "/kaggle/input"

train_images_root = images_root + "/walk_or_run_train/train"
val_images_root = images_root + "/walk_or_run_test/test"

!echo "train images classes:"
!ls {train_images_root}
!echo "train images counts:"
!find {train_images_root} -type f | wc -l

!echo ""
!echo "val images classes:"
!ls {val_images_root}
!find {val_images_root} -type f | wc -l

# change the pre-defined model for your need
pretrained_model_name = "densenet"
n_classes=2 # walk or run

full_model, head_model, preprocess = build_model(pretrained_model_name, n_classes)
print("base model use %s , classes count use %d" %(pretrained_model_name, n_classes))
print("full model :")
full_model.summary()

print("")
print("head model :")
head_model.summary()
# a simple flow dir generator wrapper
def gen(image_dir, datagen, preprocess, batch_size):
  generator = train_datagen.flow_from_directory(
        image_dir,
        batch_size=batch_size,
        target_size=(224,224),
        shuffle=True,
        class_mode='categorical')
  while True:
    X,y  = generator.next()
    if len(y) != batch_size:
      generator.reset()
      generator.on_epoch_end() # shuffle again
      continue
      
    X = preprocess(X)
    yield X, y
# start to train
batch_size = 32

train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

val_datagen = ImageDataGenerator()

train_gen = gen(train_images_root, train_datagen, preprocess, batch_size)
val_gen = gen(val_images_root, train_datagen, preprocess, batch_size)

# save the best one
checkpointer = ModelCheckpoint(filepath = pretrained_model_name+'-e{epoch:02d}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.h5', verbose=1, save_best_only=True)

history = full_model.fit_generator(train_gen, 
                                   steps_per_epoch=np.ceil(len(train)/batch_size), 
                                   nb_epoch=20, nb_worker=1, 
                                   validation_data=val_gen, 
                                   validation_steps=int(np.ceil(len(test)/batch_size)),
                                   callbacks=[checkpointer])
import matplotlib.pyplot as plt
def plot_history(history):
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], 'r')
    plt.plot(history.history['val_loss'], 'b')
    plt.subplot(1,2,2)
    plt.plot(history.history['acc'], 'r')
    plt.plot(history.history['val_acc'], 'b')
    
plot_history(history)