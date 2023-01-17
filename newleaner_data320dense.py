# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in

!pip install git+https://github.com/titu1994/keras-efficientnets.git

from keras_efficientnets import EfficientNetB4

import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from sklearn.metrics import roc_auc_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import keras

from skimage import exposure

from keras.applications.densenet import DenseNet121,DenseNet201

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg19 import VGG19

from keras.applications.resnet import ResNet50

from keras.layers import Conv2D, Activation, MaxPooling2D, MaxPool2D

from keras.layers import Flatten, Dense, BatchNormalization, Dropout, PReLU

from keras.models import Model, Input

from keras import Sequential

from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard

from keras.optimizers import SGD

import keras.backend as K

from keras import regularizers

from PIL import Image

import matplotlib.pyplot as plt

import os, random, shutil

code_path = '/kaggle/working/'
def moveFile(fileDir, tarDir, rate=0.1):

    pathDir = os.listdir(fileDir)  # 取图片的原始路径

    filenumber = len(pathDir)

    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片

    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片



    for name in sample:

        shutil.move(fileDir + name, tarDir + name)





# bn + prelu

def bn_prelu(x):

    x = BatchNormalization()(x)

    x = PReLU()(x)

    return x

def dense_model1(feat_dims, out_dims):

    dense_base = DenseNet201(include_top=False, weights=None, input_shape=(320, 320, 1))

    x = dense_base.get_layer('avg_pool', 1).output

    x = Flatten()(x)

#     fc = Dense(feat_dims)(x)

#     x = bn_prelu(fc)

#     x = Dropout(0.5)(x)

    x = Dense(out_dims)(x)

    x = Activation("softmax")(x)



    # buid myself model

    input_shape = dense_base.input

    output_shape = x



    desne121_model = Model(inputs=input_shape, outputs=output_shape)

    return desne121_model



# AUC for a binary classifier

def auc(y_true, y_pred):

    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)

    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)

    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)

    binSizes = -(pfas[1:]-pfas[:-1])

    s = ptas*binSizes

    return K.sum(s, axis=0)

#-----------------------------------------------------------------------------------------------------------------------------------------------------

# PFA, prob false alert for binary classifier

def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):

    y_pred = K.cast(y_pred >= threshold, 'float32')

    # N = total number of negative labels

    N = K.sum(1 - y_true)

    # FP = total number of false alerts, alerts from the negative class labels

    FP = K.sum(y_pred - y_pred * y_true)

    return FP/N

#-----------------------------------------------------------------------------------------------------------------------------------------------------

# P_TA prob true alerts for binary classifier

def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):

    y_pred = K.cast(y_pred >= threshold, 'float32')

    # P = total number of positive labels

    P = K.sum(y_true)

    # TP = total number of correct alerts, alerts from the positive class labels

    TP = K.sum(y_pred * y_true)

    return TP/P





def basic_model(out_dims, input_shape=(320, 320, 1)):

    inputs_dim = Input(input_shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(inputs_dim)

    x = bn_prelu(x)

    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)

    x = bn_prelu(x)

    x = MaxPool2D(pool_size=(2, 2))(x)



    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)

    x = bn_prelu(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)

    x = bn_prelu(x)

    x = MaxPool2D(pool_size=(2, 2))(x)



    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)

    x = bn_prelu(x)

    x = MaxPool2D(pool_size=(2, 2))(x)



    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)

    x = bn_prelu(x)

    x = MaxPool2D(pool_size=(2, 2))(x)



    x_flat = Flatten()(x)



    fc1 = Dense(1024)(x_flat)

    fc1 = bn_prelu(fc1)

    dp_1 = Dropout(0.3)(fc1)



    fc2 = Dense(512)(dp_1)

    fc2 = bn_prelu(fc2)

    dp_2 = Dropout(0.3)(fc2)



    fc3 = Dense(out_dims)(dp_2)

    fc2 = Activation('softmax')(fc3)



    model = Model(inputs=inputs_dim, outputs=fc2)

    return model



def dense_model(feat_dims,out_dims):

    dense_base=DenseNet121(include_top=False, weights=None, input_shape=(320, 320, 1))

    x = dense_base.get_layer('avg_pool', 1).output

    x = Flatten()(x)

    fc = Dense(feat_dims)(x)

    x = bn_prelu(fc)

    x = Dropout(0.3)(x)

    x = Dense(out_dims)(x)

    x = Activation("softmax")(x)



    # buid myself model

    input_shape = dense_base.input

    output_shape = x



    desne121_model = Model(inputs=input_shape, outputs=output_shape)

    return desne121_model

def eff_model(feat_dims,out_dims):

    dense_base=EfficientNetB4(include_top=False, weights=None, input_shape=(320, 320, 1))

    x = dense_base.get_layer('avg_pool', 1).output

    x = Flatten()(x)

    fc = Dense(feat_dims)(x)

    x = bn_prelu(fc)

    x = Dropout(0.3)(x)

    x = Dense(out_dims)(x)

    x = Activation("softmax")(x)



    # buid myself model

    input_shape = dense_base.input

    output_shape = x



    desne121_model = Model(inputs=input_shape, outputs=output_shape)

    return desne121_model



def vgg_model(feat_dims, out_dims):

    vgg_base_model = VGG19(include_top=False, weights=None, input_shape=(320, 320, 1))



    # get output of original resnet50

    x = vgg_base_model.get_layer('avg_pool', 1).output

    x = Flatten()(x)

    fc = Dense(feat_dims)(x)

    x = bn_prelu(fc)

    x = Dropout(0.5)(x)

    x = Dense(out_dims)(x)

    x = Activation("softmax")(x)



    # buid myself model

    input_shape = vgg_base_model.input

    output_shape = x



    vgg19_model = Model(inputs=input_shape, outputs=output_shape)



    return vgg19_model





def VGG16(num_classes, importModel=None):

    image_input = Input(shape=(320, 320, 1))

    # block1

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(image_input)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # block2

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # block3

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # block4

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # block5

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block

    x = Flatten(name='flatten')(x)

    x = Dense(4096, activation='relu', name='fc1')(x)

    x = Dense(4096, activation='relu', name='fc2')(x)

    x = Dense(num_classes, activation='softmax', name='fc3')(x)

    model = Model(image_input, x, name='vgg16')

    if importModel:

        model = Sequential()

        model.load_weights(importModel)

    return model





# learning rate of epoch

def lrschedule(epoch):

    if epoch <= 5:

        return 0.01

    elif epoch <= 10:

        return 0.005

    elif epoch <= 15:

        return 0.001

    elif epoch<=20:

        return 0.0001

    else:

        return 0.00001





# one-hot 2 label

def translate_onehot2label(one_hot):

    # length = num of images labels, nb_classes = classes of image

    length = one_hot.shape[0]

    nb_classes = one_hot.shape[1]



    labels = []

    for i in range(length):

        for j in range(nb_classes):

            if one_hot[i][j] == 1:

                labels.append(j)



    labels = np.array(labels).reshape((length, 1))



    return labels





# my generator for centerloss

def mygenerator(generator):

    """

    :param generator:

    :return: x: [x, y_value], y: [y, random_centers]

    """



    while True:

        data = next(generator)

        x, y = data[0], data[1]

        # not one-hot encoding

        y_value = translate_onehot2label(y)



        random_centers = np.random.randn(BATCH_SIZE, 1)



        data_x = [x, y_value]

        data_y = [y, random_centers]

        yield data_x, data_y





# training model

def model_train(model, loadweights, isCenterloss, lambda_center):

    lr = LearningRateScheduler(lrschedule)

    mdcheck = ModelCheckpoint(WEIGHTS_PATH, monitor='val_acc', save_best_only=True)

    if loadweights:

        if os.path.isfile(WEIGHTS_PATH):

            model.load_weights(WEIGHTS_PATH)

            print('model have load pre weights!!')

        else:

            print('model not load weights!!')

    else:

        print('not load weights model')



    # optimizer use sgd

    adam = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    if not isCenterloss:

        # common cnn model

        print("model compile!!")

        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[auc])

        print("model training!!")

        history = model.fit_generator(train_generator,

                                      steps_per_epoch=40126// BATCH_SIZE,

                                      epochs=max_Epochs,

                                      validation_data=val_generator,

                                      validation_steps=6359 // BATCH_SIZE,

                                      callbacks=[lr, mdcheck])



    return history



    # draw and save loss pic and acc pic











def label_of_directory(directory):

    """

    sorted for label indices

    return a dict for {'classes', 'range(len(classes))'}

    """

    classes = []

    for subdir in sorted(os.listdir(directory)):

        if os.path.isdir(os.path.join(directory, subdir)):

            classes.append(subdir)



    num_classes = len(classes)

    class_indices = dict(zip(classes, range(len(classes))))

    return class_indices



    # get key from value in dict





def get_key_from_value(dict, index):

    for keys, values in dict.items():

        if values == index:

            return keys



    # geneartor list of image list in test





def generator_list_of_imagepath(path):

    image_list = []

    for image in os.listdir(path):

        if not image == '.DS_Store':

            image_list.append(path + image)

    return image_list



    # read image and resize to gray





def load_image(image):

    img = Image.open(image)

    img = img.resize((320, 320))

    img=exposure.equalize_adapthist(img)

    img = np.array(img)

    img = img / 255

    img = img.reshape((1,) + img.shape + (1,))  # reshape img to size(1, 128, 128, 1)

    return img
train_path = '/kaggle/input/finadata320/train/'

val_path = '/kaggle/input/finadata320/val/'

num_classes = 2

feat_dims = 128

BATCH_SIZE = 64

WEIGHTS_PATH = 'best_weights_hanzi.hdf5'

max_Epochs = 40



train_datagen = ImageDataGenerator(

    horizontal_flip=True,

    rescale=1. / 255,

    )



val_datagen = ImageDataGenerator(

    rescale=1. / 255

)



train_generator = train_datagen.flow_from_directory(

    train_path,

    target_size=(320, 320),

    batch_size=BATCH_SIZE,

    color_mode='grayscale',

    class_mode='categorical'

)

val_generator = val_datagen.flow_from_directory(

    val_path,

    target_size=(320, 320),

    batch_size=BATCH_SIZE,

    color_mode='grayscale',

    class_mode='categorical'

    )

#simple_model=eff_model(256,2)

#simple_model = VGG16(2)

simple_model=basic_model(2)

print(simple_model.summary())

print("=====start train image of epoch=====")



model_history = model_train(simple_model, False, isCenterloss=False, lambda_center=0.01)



print("=====show acc and loss of train and val====")



def draw_loss_acc(history):

    x_trick = [x + 1 for x in range(max_Epochs)]

    loss = history.history['loss']

    acc = history.history['accuracy']

    val_loss = history.history['val_loss']

    val_acc = history.history['val_accuracy']



    plt.style.use('ggplot')



    plt.figure(figsize=(10, 6))

    plt.title('model = %s, batch_size = %s' % ('losses', BATCH_SIZE))

    plt.plot(x_trick, loss, 'g-', label='loss')

    plt.plot(x_trick, val_loss, 'y-', label='val_loss')

    plt.legend()

    plt.xlabel('epochs')

    plt.ylabel('loss')

    plt.show()

    plt.savefig(code_path + 'loss.png', format='png', dpi=300)



    plt.figure(figsize=(10, 6))

    plt.title('learninngRate = %s, batch_size = %s' % ('accuracy', BATCH_SIZE))

    plt.plot(x_trick, val_acc, 'y-', label='val_acc')

    plt.plot(x_trick, acc, 'b-', label='acc')

    plt.legend()

    plt.xlabel('epochs')

    plt.ylabel('acc')

    plt.show()

    plt.savefig(code_path + 'acc.png', format='png', dpi=300)
draw_loss_acc(model_history)



print("====done!=====")
