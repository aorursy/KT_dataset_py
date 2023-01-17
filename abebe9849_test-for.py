# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



load_dir = '../input/grapheme-imgs-128x128/'



train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

train['filename'] = train.image_id.apply(lambda filename: load_dir + filename + '.png')



train.head()
import matplotlib.pyplot as plt



import os, time, random, glob, pickle, warnings, math



from keras import backend as K

import tensorflow as tf

from keras.backend.tensorflow_backend import set_session

from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.applications.resnet50 import  ResNet50

from keras.applications.densenet import  DenseNet121

from keras.layers import GlobalMaxPooling2D,Conv2D, Dense,BatchNormalization,Flatten,Input, Multiply ,Dropout,GlobalAveragePooling2D

from keras.utils import Sequence

from keras.optimizers import Adam

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

import cv2

warnings.filterwarnings('ignore') # warningを非表示にする

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # kerasのwarningを非表示にする
from __future__ import print_function

from __future__ import absolute_import



import warnings



from keras.layers import Input

from keras import layers

from keras.layers import Dense

from keras.layers import Activation

from keras.layers import Flatten

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import AveragePooling2D

from keras.layers import GlobalAveragePooling2D

from keras.layers import GlobalMaxPooling2D

from keras.layers import BatchNormalization

from keras.layers import Reshape

from keras.layers import Multiply

from keras.models import Model

from keras import backend as K

from keras.engine.topology import get_source_inputs

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

#from keras.applications.imagenet_utils import _obtain_input_shape





def preprocess_input(x):

    # 'RGB'->'BGR'

    x = x[..., ::-1]

    

    # Zero-center by mean pixel

    x[..., 0] -= 103.939

    x[..., 1] -= 116.779

    x[..., 2] -= 123.68



    # Scale

    x *= 0.017

    return x

  

def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters

    if K.image_data_format() == 'channels_last':

        bn_axis = 3

    else:

        bn_axis = 1

    bn_eps = 0.0001

        

    block_name = str(stage) + "_" + str(block)

    conv_name_base = "conv" + block_name

    relu_name_base = "relu" + block_name



    x = Conv2D(filters1, (1, 1), use_bias=False, name=conv_name_base + '_x1')(input_tensor)

    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x1_bn')(x)

    x = Activation('relu', name=relu_name_base + '_x1')(x)



    x = Conv2D(filters2, kernel_size, padding='same', use_bias=False, name=conv_name_base + '_x2')(x)

    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x2_bn')(x)

    x = Activation('relu', name=relu_name_base + '_x2')(x)



    x = Conv2D(filters3, (1, 1), use_bias=False, name=conv_name_base + '_x3')(x)

    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x3_bn')(x)



    se = GlobalAveragePooling2D(name='pool' + block_name + '_gap')(x)

    se = Dense(filters3 // 16, activation='relu', name = 'fc' + block_name + '_sqz')(se)

    se = Dense(filters3, activation='sigmoid', name = 'fc' + block_name + '_exc')(se)

    se = Reshape([1, 1, filters3])(se)

    x = Multiply(name='scale' + block_name)([x, se])



    x = layers.add([x, input_tensor], name='block_' + block_name)

    x = Activation('relu', name=relu_name_base)(x)

    return x





def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    if K.image_data_format() == 'channels_last':

        bn_axis = 3

    else:

        bn_axis = 1

    bn_eps = 0.0001

    

    block_name = str(stage) + "_" + str(block)

    conv_name_base = "conv" + block_name

    relu_name_base = "relu" + block_name



    x = Conv2D(filters1, (1, 1), use_bias=False, name=conv_name_base + '_x1')(input_tensor)

    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x1_bn')(x)

    x = Activation('relu', name=relu_name_base + '_x1')(x)



    x = Conv2D(filters2, kernel_size, strides=strides, padding='same', use_bias=False, name=conv_name_base + '_x2')(x)

    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x2_bn')(x)

    x = Activation('relu', name=relu_name_base + '_x2')(x)



    x = Conv2D(filters3, (1, 1), use_bias=False, name=conv_name_base + '_x3')(x)

    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x3_bn')(x)

    

    se = GlobalAveragePooling2D(name='pool' + block_name + '_gap')(x)

    se = Dense(filters3 // 16, activation='relu', name = 'fc' + block_name + '_sqz')(se)

    se = Dense(filters3, activation='sigmoid', name = 'fc' + block_name + '_exc')(se)

    se = Reshape([1, 1, filters3])(se)

    x = Multiply(name='scale' + block_name)([x, se])

    

    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=False, name=conv_name_base + '_prj')(input_tensor)

    shortcut = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_prj_bn')(shortcut)



    x = layers.add([x, shortcut], name='block_' + block_name)

    x = Activation('relu', name=relu_name_base)(x)

    return x





def SEResNet50(include_top=True, weights='imagenet',

               input_tensor=None, input_shape=(125,125,3),

               pooling=None,

               classes=1000):



    # Determine proper input shape

    input_shape=(125,125,3)



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

    bn_eps = 0.0001



    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', use_bias=False, name='conv1')(img_input)

    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name='conv1_bn')(x)

    x = Activation('relu', name='relu1')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)



    x = conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))

    x = identity_block(x, 3, [64, 64, 256], stage=2, block=2)

    x = identity_block(x, 3, [64, 64, 256], stage=2, block=3)



    x = conv_block(x, 3, [128, 128, 512], stage=3, block=1)

    x = identity_block(x, 3, [128, 128, 512], stage=3, block=2)

    x = identity_block(x, 3, [128, 128, 512], stage=3, block=3)

    x = identity_block(x, 3, [128, 128, 512], stage=3, block=4)



    x = conv_block(x, 3, [256, 256, 1024], stage=4, block=1)

    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=2)

    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=3)

    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=4)

    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=5)

    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=6)



    x = conv_block(x, 3, [512, 512, 2048], stage=5, block=1)

    x = identity_block(x, 3, [512, 512, 2048], stage=5, block=2)

    x = identity_block(x, 3, [512, 512, 2048], stage=5, block=3)





    # Ensure that the model takes into account

    # any potential predecessors of `input_tensor`.

    if input_tensor is not None:

        inputs = get_source_inputs(input_tensor)

    else:

        inputs = img_input

    # Create model.

    model = Model(inputs, x, name='se-resnet50')

    return model  

  

  

se_resnet50 = SEResNet50(weights=None, input_shape=(125, 125, 3),include_top=False)
!pip install ../input/efficientnet-keras-source-code/repository/qubvel-efficientnet-c993591

import efficientnet.keras as efn 

from keras.applications.resnet50 import ResNet50

# we will be using EfficientNetB0

wg0 = '../input/efficientnet-keras-weights-b0b5/efficientnet-b0_imagenet_1000_notop.h5'

wg1 = '../input/efficientnet-keras-weights-b0b5/efficientnet-b1_imagenet_1000_notop.h5'

wg2 = '../input/efficientnet-keras-weights-b0b5/efficientnet-b2_imagenet_1000_notop.h5'

wg3 = '../input/efficientnet-keras-weights-b0b5/efficientnet-b3_imagenet_1000_notop.h5'

#wg_res="../input/resnet-50"

efnet_b0 = efn.EfficientNetB0(weights=wg0, include_top = False, input_shape=(125, 125, 3))

efnet_b1 = efn.EfficientNetB1(weights=wg1, include_top = False, input_shape=(125, 125, 3))

efnet_b2 = efn.EfficientNetB2(weights=wg2, include_top = False, input_shape=(125, 125, 3))

efnet_b3 = efn.EfficientNetB3(weights=wg3, include_top = False, input_shape=(125, 125, 3))



print(tf.keras.__version__)

print(tf.__version__)
# code: https://github.com/titu1994/keras-adabound   

class AdaBound(optimizers.Optimizer):

    """AdaBound optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments

        lr: float >= 0. Learning rate.

        final_lr: float >= 0. Final learning rate.

        beta_1: float, 0 < beta < 1. Generally close to 1.

        beta_2: float, 0 < beta < 1. Generally close to 1.

        gamma: float >= 0. Convergence speed of the bound function.

        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.

        decay: float >= 0. Learning rate decay over each update.

        weight_decay: Weight decay weight.

        amsbound: boolean. Whether to apply the AMSBound variant of this

            algorithm.

    # References

        - [Adaptive Gradient Methods with Dynamic Bound of Learning Rate]

          (https://openreview.net/forum?id=Bkg3g2R9FX)

        - [Adam - A Method for Stochastic Optimization]

          (https://arxiv.org/abs/1412.6980v8)

        - [On the Convergence of Adam and Beyond]

          (https://openreview.net/forum?id=ryQu7f-RZ)

    """



    def __init__(self, learning_rate=0.001, final_lr=0.1, beta_1=0.9, beta_2=0.999, gamma=1e-3,

                 epsilon=None, decay=0., amsbound=False, weight_decay=0.0, **kwargs):

        super(AdaBound, self).__init__(**kwargs)



        if not 0. <= gamma <= 1.:

            raise ValueError("Invalid `gamma` parameter. Must lie in [0, 1] range.")



        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.learning_rate = K.variable(learning_rate, name='learning_rate')

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')



        self.final_lr = final_lr

        self.gamma = gamma



        if epsilon is None:

            epsilon = K.epsilon()

        self.epsilon = epsilon

        self.initial_decay = decay

        self.amsbound = amsbound



        self.weight_decay = float(weight_decay)

        self.base_lr = float(learning_rate)



    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]



        learning_rate = self.learning_rate

        if self.initial_decay > 0:

            learning_rate = learning_rate * (1. / (1. + self.decay * K.cast(self.iterations,

                                                      K.dtype(self.decay))))



        t = K.cast(self.iterations, K.floatx()) + 1



        # Applies bounds on actual learning rate

        step_size = learning_rate * (K.sqrt(1. - K.pow(self.beta_2, t)) /

                          (1. - K.pow(self.beta_1, t)))



        final_lr = self.final_lr * learning_rate / self.base_lr

        lower_bound = final_lr * (1. - 1. / (self.gamma * t + 1.))

        upper_bound = final_lr * (1. + 1. / (self.gamma * t))



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsbound:

            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        else:

            vhats = [K.zeros(1) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats



        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):

            # apply weight decay

            if self.weight_decay != 0.:

                g += self.weight_decay * K.stop_gradient(p)



            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)



            if self.amsbound:

                vhat_t = K.maximum(vhat, v_t)

                denom = (K.sqrt(vhat_t) + self.epsilon)

                self.updates.append(K.update(vhat, vhat_t))

            else:

                denom = (K.sqrt(v_t) + self.epsilon)



            # Compute the bounds

            step_size_p = step_size * K.ones_like(denom)

            step_size_p_bound = step_size_p / denom

            bounded_lr_t = m_t * K.minimum(K.maximum(step_size_p_bound,

                                                     lower_bound), upper_bound)



            p_t = p - bounded_lr_t



            self.updates.append(K.update(m, m_t))

            self.updates.append(K.update(v, v_t))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, new_p))

        return self.updates



    def get_config(self):

        config = {'learning_rate': float(K.get_value(self.learning_rate)),

                  'final_lr': float(self.final_lr),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'gamma': float(self.gamma),

                  'decay': float(K.get_value(self.decay)),

                  'epsilon': self.epsilon,

                  'weight_decay': self.weight_decay,

                  'amsbound': self.amsbound}

        base_config = super(AdaBound, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
def get_model(base_model):   

    input_=Input((125,125,1))

    x = Conv2D(3, (3, 3), padding='same')(input_)

    base_model=base_model(x)

    x = BatchNormalization()(base_model)

    x = Dropout(0.5)(x)

    x=Flatten()(x)

    x = Dense(256, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    out_grapheme = Dense(168, activation='softmax', name='grapheme')(x)

    out_vowel = Dense(11, activation='softmax', name='vowel')(x)

    out_consonant = Dense(7, activation='softmax', name='consonant')(x)

    

    return Model(inputs=input_, outputs=[out_grapheme, out_vowel, out_consonant])

model_se_resnet=get_model(se_resnet50)

model_se_resnet.summary()

model_se_resnet.compile(AdaBound(), metrics=['accuracy'], loss='sparse_categorical_crossentropy')



def get_pad_width(im, new_shape, is_rgb=True):

    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]

    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)

    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)

    if is_rgb:

        pad_width = ((t,b), (l,r), (0, 0))

    else:

        pad_width = ((t,b), (l,r))

    return pad_width
print(cv2.__version__)
def crop_object(file, thresh=220, maxval=255, square=True):

    """

    Source: https://stackoverflow.com/questions/49577973/how-to-crop-the-biggest-object-in-image-with-python-opencv

    """

    img=cv2.imread(file)

    gray = cv2.imread(file,cv2.IMREAD_GRAYSCALE) # convert to grayscale

    # threshold to get just the signature (INVERTED)

    retval, thresh_gray = cv2.threshold(gray, thresh=thresh, maxval=maxval, type=cv2.THRESH_BINARY_INV)



    contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



    # Find object with the biggest bounding box

    mx = (0,0,0,0)      # biggest bounding box so far

    mx_area = 0

    for cont in contours:

        x,y,w,h = cv2.boundingRect(cont)

        area = w*h

        if area > mx_area:

            mx = x,y,w,h

            mx_area = area

    x,y,w,h = mx

    

    crop = img[y:y+h, x:x+w]

    

    if square:

        pad_width = get_pad_width(crop, max(crop.shape))

        crop = np.pad(crop, pad_width=pad_width, mode='constant', constant_values=255)

    

    return crop
def data_generator(filenames, y, batch_size=64, shape=(125, 125, 1), random_state=2019):

    y = y.copy()

    np.random.seed(random_state)

    indices = np.arange(len(filenames))

    

    while True:

        np.random.shuffle(indices)

        

        for i in range(0, len(indices), batch_size):

            batch_idx = indices[i:i+batch_size]

            size = len(batch_idx)

            

            batch_files = filenames[batch_idx]

            X_batch = np.zeros((size, *shape))

            y_batch = y[batch_idx]

            

            for i, file in enumerate(batch_files):

                img = crop_object(file, thresh=220)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                img = cv2.resize(img, shape[:2])

                X_batch[i, :, :, 0] = img / 255.

            

            yield X_batch, [y_batch[:, i] for i in range(y_batch.shape[1])]
tra_files=train.filename.values[:100]
#test

thresh=220

maxval=255

file=tra_files[0]

img=cv2.imread(file)

print(img.shape)

gray = cv2.imread(file,cv2.IMREAD_GRAYSCALE) # convert to grayscale

    # threshold to get just the signature (INVERTED) 

retval, thresh_gray = cv2.threshold(gray, thresh=thresh, maxval=maxval, type=cv2.THRESH_BINARY_INV)



contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



    # Find object with the biggest bounding box

mx = (0,0,0,0)      # biggest bounding box so far

mx_area = 0

for cont in contours:

    x,y,w,h = cv2.boundingRect(cont)

    area = w*h

    if area > mx_area:

        mx = x,y,w,h

        mx_area = area

x,y,w,h = mx

crop = img[y:y+h, x:x+w]

shape=(125, 125, 1)

img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

img = cv2.resize(img, shape[:2])

print(img.shape)
from sklearn.model_selection import train_test_split

import cv2

train_files, valid_files, y_train, y_valid = train_test_split(

    train.filename.values, 

    train[['grapheme_root','vowel_diacritic', 'consonant_diacritic']].values, 

    test_size=0.25, 

    random_state=2019

)
batch_size = 128



train_gen = data_generator(train_files, y_train)

valid_gen = data_generator(valid_files, y_valid)



train_steps = round(len(train_files) / batch_size) + 1

valid_steps = round(len(valid_files) / batch_size) + 1
callbacks = [tf.keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True)]



train_history = model_se_resnet.fit_generator(

    train_gen,

    steps_per_epoch=train_steps,

    epochs=1,

    validation_data=valid_gen,

    validation_steps=valid_steps,

    #callbacks=callbacks

).history
SEED = 2020

batch_size = 12 

dim = (125, 125)

SIZE = 125

stats = (0.0692, 0.2051)

HEIGHT = 137 

WIDTH = 236

from tqdm import tqdm

def bbox(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax



def crop_resize(img0, size=125, pad=16):

    #crop a box around pixels large than the threshold 

    #some images contain line at the sides

    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)

    #cropping may cut too much, so we need to add it back

    xmin = xmin - 13 if (xmin > 13) else 0

    ymin = ymin - 10 if (ymin > 10) else 0

    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH

    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT

    img = img0[ymin:ymax,xmin:xmax]

    #remove lo intensity pixels as noise

    img[img < 28] = 0

    lx, ly = xmax-xmin,ymax-ymin

    l = max(lx,ly) + pad

    #make sure that the aspect ratio is kept in rescaling

    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    

    return cv2.resize(img,(size,size))
import gc

def test_batch_generator(df, batch_size):

    num_imgs = len(df)



    for batch_start in range(0, num_imgs, batch_size):

        curr_batch_size = min(num_imgs, batch_start + batch_size) - batch_start

        idx = np.arange(batch_start, batch_start + curr_batch_size)



        names_batch = df.iloc[idx, 0].values

        imgs_batch = 255 - df.iloc[idx, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

        X_batch = np.zeros((curr_batch_size, SIZE, SIZE, 1))

        

        for j in range(curr_batch_size):

            img = (imgs_batch[j,]*(255.0/imgs_batch[j,].max())).astype(np.uint8)

            img = crop_resize(img, size=SIZE)

            img = img[:, :, np.newaxis]

            X_batch[j,] = img



        yield X_batch, names_batch

TEST = [

    "../input/bengaliai-cv19/test_image_data_0.parquet",

    "../input/bengaliai-cv19/test_image_data_1.parquet",

    "../input/bengaliai-cv19/test_image_data_2.parquet",

    "../input/bengaliai-cv19/test_image_data_3.parquet",

]



# placeholders 

row_id = []

target = []



# iterative over the test sets

for fname in tqdm(TEST):

    test_ = pd.read_parquet(fname)

    test_gen = test_batch_generator(test_, batch_size=batch_size)



    for batch_x, batch_name in test_gen:

        batch_predict = model_se_resnet.predict(batch_x)

        for idx, name in enumerate(batch_name):

            row_id += [

                f"{name}_consonant_diacritic",

                f"{name}_grapheme_root",

                f"{name}_vowel_diacritic",

            ]

            target += [

                np.argmax(batch_predict[2], axis=1)[idx],

                np.argmax(batch_predict[0], axis=1)[idx],

                np.argmax(batch_predict[1], axis=1)[idx],

            ]



    del test_

    gc.collect()

    

    

df_sample = pd.DataFrame(

    {

        'row_id': row_id,

        'target':target

    },

    columns = ['row_id','target'] 

)



df_sample.to_csv('submission.csv',index=False)

gc.collect()