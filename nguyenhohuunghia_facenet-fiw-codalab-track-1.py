import numpy as np

import os

import matplotlib.pyplot as plt

import cv2

from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import proj3d

from imageio import imread

from skimage.transform import resize

from scipy.spatial import distance

from keras.models import load_model

import pandas as pd

from tqdm import tqdm
!pip install git+https://github.com/rcmalli/keras-vggface.git
from collections import defaultdict

from glob import glob

from random import choice, sample



from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract

from keras.models import Model

from keras.optimizers import Adam

from keras_vggface.utils import preprocess_input
train_file_path = "../input/fiwcodalab/train-pairs-updated.pkl"

train_folders_path = "../input/fiwcodalab/train-faces/train-faces/"

val_famillies = "F0009"



all_famillies = glob(train_folders_path + "*/")

print(len(all_famillies))

all_images = glob(train_folders_path + "*/*/*.jpg")

print('Number of all images', len(all_images))



train_images = [x for x in all_images if val_famillies not in x]

val_images = [x for x in all_images if val_famillies in x]





train_person_to_images_map = defaultdict(list)



ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]



for x in train_images:

    train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)



val_person_to_images_map = defaultdict(list)



for x in val_images:

    val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)



relationships = pd.read_pickle(train_file_path)

print(relationships)

relationships = list(zip(relationships.p1.values, relationships.p2.values))

relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]



train = [x for x in relationships if val_famillies not in x[0]]

val = [x for x in relationships if val_famillies in x[0]]
print(len(train), len(val))
def read_img(path):

    img = cv2.imread(path)

    img = cv2.resize(img, (160, 160))

    img = np.array(img).astype(np.float)

    return preprocess_input(img, version=2)





def gen(list_tuples, person_to_images_map, batch_size=16):

    ppl = list(person_to_images_map.keys())

    while True:

        batch_tuples = sample(list_tuples, batch_size // 2)

        labels = [1] * len(batch_tuples)

        while len(batch_tuples) < batch_size:

            p1 = choice(ppl)

            p2 = choice(ppl)



            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:

                batch_tuples.append((p1, p2))

                labels.append(0)



        for x in batch_tuples:

            if not len(person_to_images_map[x[0]]):

                print(x[0])



        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]

        X1 = np.array([read_img(x) for x in X1])



        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]

        X2 = np.array([read_img(x) for x in X2])



        yield [X1, X2], labels

"""Inception-ResNet V1 model for Keras.

# Reference

http://arxiv.org/abs/1602.07261

https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py

https://github.com/myutwo150/keras-inception-resnet-v2/blob/master/inception_resnet_v2.py

"""

from functools import partial



from keras.models import Model

from keras.layers import Activation

from keras.layers import BatchNormalization

from keras.layers import Concatenate

from keras.layers import Conv2D

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import GlobalAveragePooling2D

from keras.layers import Input

from keras.layers import Lambda

from keras.layers import MaxPooling2D

from keras.layers import add

from keras import backend as K





def scaling(x, scale):

    return x * scale





def conv2d_bn(x,

              filters,

              kernel_size,

              strides=1,

              padding='same',

              activation='relu',

              use_bias=False,

              name=None):

    x = Conv2D(filters,

               kernel_size,

               strides=strides,

               padding=padding,

               use_bias=use_bias,

               name=name)(x)

    if not use_bias:

        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3

        bn_name = _generate_layer_name('BatchNorm', prefix=name)

        x = BatchNormalization(axis=bn_axis, momentum=0.995, epsilon=0.001,

                               scale=False, name=bn_name)(x)

    if activation is not None:

        ac_name = _generate_layer_name('Activation', prefix=name)

        x = Activation(activation, name=ac_name)(x)

    return x





def _generate_layer_name(name, branch_idx=None, prefix=None):

    if prefix is None:

        return None

    if branch_idx is None:

        return '_'.join((prefix, name))

    return '_'.join((prefix, 'Branch', str(branch_idx), name))





def _inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3

    if block_idx is None:

        prefix = None

    else:

        prefix = '_'.join((block_type, str(block_idx)))

    name_fmt = partial(_generate_layer_name, prefix=prefix)



    if block_type == 'Block35':

        branch_0 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_1x1', 0))

        branch_1 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 1))

        branch_1 = conv2d_bn(branch_1, 32, 3, name=name_fmt('Conv2d_0b_3x3', 1))

        branch_2 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 2))

        branch_2 = conv2d_bn(branch_2, 32, 3, name=name_fmt('Conv2d_0b_3x3', 2))

        branch_2 = conv2d_bn(branch_2, 32, 3, name=name_fmt('Conv2d_0c_3x3', 2))

        branches = [branch_0, branch_1, branch_2]

    elif block_type == 'Block17':

        branch_0 = conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_1x1', 0))

        branch_1 = conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_0a_1x1', 1))

        branch_1 = conv2d_bn(branch_1, 128, [1, 7], name=name_fmt('Conv2d_0b_1x7', 1))

        branch_1 = conv2d_bn(branch_1, 128, [7, 1], name=name_fmt('Conv2d_0c_7x1', 1))

        branches = [branch_0, branch_1]

    elif block_type == 'Block8':

        branch_0 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_1x1', 0))

        branch_1 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))

        branch_1 = conv2d_bn(branch_1, 192, [1, 3], name=name_fmt('Conv2d_0b_1x3', 1))

        branch_1 = conv2d_bn(branch_1, 192, [3, 1], name=name_fmt('Conv2d_0c_3x1', 1))

        branches = [branch_0, branch_1]

    else:

        raise ValueError('Unknown Inception-ResNet block type. '

                         'Expects "Block35", "Block17" or "Block8", '

                         'but got: ' + str(block_type))



    mixed = Concatenate(axis=channel_axis, name=name_fmt('Concatenate'))(branches)

    up = conv2d_bn(mixed,

                   K.int_shape(x)[channel_axis],

                   1,

                   activation=None,

                   use_bias=True,

                   name=name_fmt('Conv2d_1x1'))

    up = Lambda(scaling,

                output_shape=K.int_shape(up)[1:],

                arguments={'scale': scale})(up)

    x = add([x, up])

    if activation is not None:

        x = Activation(activation, name=name_fmt('Activation'))(x)

    return x





def InceptionResNetV1(input_shape=(160, 160, 3),

                      classes=128,

                      dropout_keep_prob=0.8,

                      weights_path=None):

    inputs = Input(shape=input_shape)

    x = conv2d_bn(inputs, 32, 3, strides=2, padding='valid', name='Conv2d_1a_3x3')

    x = conv2d_bn(x, 32, 3, padding='valid', name='Conv2d_2a_3x3')

    x = conv2d_bn(x, 64, 3, name='Conv2d_2b_3x3')

    x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)

    x = conv2d_bn(x, 80, 1, padding='valid', name='Conv2d_3b_1x1')

    x = conv2d_bn(x, 192, 3, padding='valid', name='Conv2d_4a_3x3')

    x = conv2d_bn(x, 256, 3, strides=2, padding='valid', name='Conv2d_4b_3x3')



    # 5x Block35 (Inception-ResNet-A block):

    for block_idx in range(1, 6):

        x = _inception_resnet_block(x,

                                    scale=0.17,

                                    block_type='Block35',

                                    block_idx=block_idx)



    # Mixed 6a (Reduction-A block):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3

    name_fmt = partial(_generate_layer_name, prefix='Mixed_6a')

    branch_0 = conv2d_bn(x,

                         384,

                         3,

                         strides=2,

                         padding='valid',

                         name=name_fmt('Conv2d_1a_3x3', 0))

    branch_1 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))

    branch_1 = conv2d_bn(branch_1, 192, 3, name=name_fmt('Conv2d_0b_3x3', 1))

    branch_1 = conv2d_bn(branch_1,

                         256,

                         3,

                         strides=2,

                         padding='valid',

                         name=name_fmt('Conv2d_1a_3x3', 1))

    branch_pool = MaxPooling2D(3,

                               strides=2,

                               padding='valid',

                               name=name_fmt('MaxPool_1a_3x3', 2))(x)

    branches = [branch_0, branch_1, branch_pool]

    x = Concatenate(axis=channel_axis, name='Mixed_6a')(branches)



    # 10x Block17 (Inception-ResNet-B block):

    for block_idx in range(1, 11):

        x = _inception_resnet_block(x,

                                    scale=0.1,

                                    block_type='Block17',

                                    block_idx=block_idx)



    # Mixed 7a (Reduction-B block): 8 x 8 x 2080

    name_fmt = partial(_generate_layer_name, prefix='Mixed_7a')

    branch_0 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 0))

    branch_0 = conv2d_bn(branch_0,

                         384,

                         3,

                         strides=2,

                         padding='valid',

                         name=name_fmt('Conv2d_1a_3x3', 0))

    branch_1 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 1))

    branch_1 = conv2d_bn(branch_1,

                         256,

                         3,

                         strides=2,

                         padding='valid',

                         name=name_fmt('Conv2d_1a_3x3', 1))

    branch_2 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 2))

    branch_2 = conv2d_bn(branch_2, 256, 3, name=name_fmt('Conv2d_0b_3x3', 2))

    branch_2 = conv2d_bn(branch_2,

                         256,

                         3,

                         strides=2,

                         padding='valid',

                         name=name_fmt('Conv2d_1a_3x3', 2))

    branch_pool = MaxPooling2D(3,

                               strides=2,

                               padding='valid',

                               name=name_fmt('MaxPool_1a_3x3', 3))(x)

    branches = [branch_0, branch_1, branch_2, branch_pool]

    x = Concatenate(axis=channel_axis, name='Mixed_7a')(branches)



    # 5x Block8 (Inception-ResNet-C block):

    for block_idx in range(1, 6):

        x = _inception_resnet_block(x,

                                    scale=0.2,

                                    block_type='Block8',

                                    block_idx=block_idx)

    x = _inception_resnet_block(x,

                                scale=1.,

                                activation=None,

                                block_type='Block8',

                                block_idx=6)



    # Classification block

    x = GlobalAveragePooling2D(name='AvgPool')(x)

    x = Dropout(1.0 - dropout_keep_prob, name='Dropout')(x)

    # Bottleneck

    x = Dense(classes, use_bias=False, name='Bottleneck')(x)

    bn_name = _generate_layer_name('BatchNorm', prefix='Bottleneck')

    x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False,

                           name=bn_name)(x)



    # Create model

    model = Model(inputs, x, name='inception_resnet_v1')

    if weights_path is not None:

        model.load_weights(weights_path)



    return model
from keras.models import Sequential



model_path = '../input/facenet/keras-facenet/model/facenet_keras.h5'

weights_path = '../input/facenet/keras-facenet/weights/facenet_keras_weights.h5'

# model = load_model(model_path)

# model.summary()



def baseline_model(img_size=160):

    input_1 = Input(shape=(img_size, img_size, 3))

    input_2 = Input(shape=(img_size, img_size, 3))



#     base_model = VGGFace(model='resnet50', include_top=False, pooling='avg')

#     base_model = load_model(model_path)

    base_model = InceptionResNetV1(weights_path=weights_path)

    

#     model = Sequential()

#     for layer in base_model.layers[1:]: # skip the first layer

#         model.add(layer)

#     model.summary()

    



    for x in base_model.layers[:-3]:

        x.trainable = True



    x1 = base_model(input_1)

    x2 = base_model(input_2)



#     x1_ = Reshape(target_shape=(7*7, 2048))(x1)

#     x2_ = Reshape(target_shape=(7*7, 2048))(x2)

    

#     x_dot = Dot(axes=[2, 2], normalize=True)([x1_, x2_])

#     x_dot = Flatten()(x_dot)



#     x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])

#     x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])



    x3 = Subtract()([x1, x2])

    x3 = Multiply()([x3, x3])



    x = Multiply()([x1, x2])



    x = Concatenate(axis=-1)([x, x3])



    x = Dense(100, activation="relu")(x)

    x = Dropout(0.01)(x)

    output = Dense(1, activation="sigmoid")(x)



    model = Model([input_1, input_2], output)



    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))



#     model.summary()



    return model
# # for training the model

# file_path = './facenet_weights.h5'



# checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')



# reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)



# callbacks_list = [checkpoint, reduce_on_plateau]



# model = baseline_model()

# # model.load_weights(file_path)

# model.fit_generator(gen(train, train_person_to_images_map, batch_size=16), use_multiprocessing=True,

#                     validation_data=gen(val, val_person_to_images_map, batch_size=16), epochs=100, verbose=2,

#                     workers=4, callbacks=callbacks_list, steps_per_epoch=200, validation_steps=100)



# test_path = "../input/fiwcodalab/test-faces/test-faces/"
weights_path = '../input/fiwcodalab/facenet-finetune/final_facenet_weights.h5'

new_model = load_model(weights_path)

new_model.summary()
len(new_model.layers)
# model.save('final_facenet_weights.h5')
def chunker(seq, size=32):

    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

from tqdm import tqdm



submission = pd.read_excel('../input/fiwcodalab/test_competition.xlsx').set_index(index)

print("The number of files is ", len(submission))

predictions = []
submission['img_pair'] = submission['p1'] + '-' + submission['p2']

# submission['img_pairs'] = submission['p1'][-5:]

print(submission.head())
test_path = '../input/fiwcodalab/test-faces/test-faces/'

for batch in tqdm(chunker(submission.img_pair.values)):

    X1 = [x.split("-")[0] for x in batch]

#     [print(test_path + x) for x in X1]

    X1 = np.array([read_img(test_path + x) for x in X1])



    X2 = [x.split("-")[1] for x in batch]

    X2 = np.array([read_img(test_path + x) for x in X2])



#     X1_embds = new_model.predict_on_batch(X1)

#     X2_embds = new_model.predict_on_batch(X2)

    

    pred = new_model.predict_on_batch([X1, X2]).ravel().tolist()

    pred = map(round, pred)

    predictions += pred



submission['label'] = predictions



submission['label'].to_csv("predictions.csv", header=True)


