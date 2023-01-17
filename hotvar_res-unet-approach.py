import os, shutil

import numpy as np

from PIL import Image

import pandas as pd

import matplotlib.pyplot as plt

import random

from glob import glob

from tqdm import tqdm_notebook

from keras.models import *

from keras.layers import *

from keras import losses 

from keras.utils import Sequence

from keras.optimizers import *

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from keras.preprocessing.image import ImageDataGenerator

import keras.backend as K

from sklearn.model_selection import train_test_split

from IPython.display import clear_output
PATH_TRAIN = os.path.join('../input/severstal-steel-defect-detection/train_images/')

PATH_TEST = os.path.join('../input/severstal-steel-defect-detection/test_images/')

file_train = glob(os.path.join(PATH_TRAIN, '*.jpg'))

file_test = sorted(glob(os.path.join(PATH_TEST, '*.jpg')))

LEN_TRAIN = len(file_train)

LEN_TEST = len(file_test)

df_rle = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

print(f'length - train data : {LEN_TRAIN}')

print(f'length - test data: {LEN_TEST}')
# check image size identity

tmp = Image.open(file_train[0])

size = tmp.size

is_identical = True

print('check size identity...')

for fname in tqdm_notebook(file_train+file_test):

    img = Image.open(fname)

    if size != img.size:

        print('found abnormal size!!')

        is_identical = False

        break

if is_identical:

    print(f'all images are {size}')

    W = size[0]

    H = size[1]
counts = (df_rle['EncodedPixels'].isnull()).value_counts(ascending=True)

plt.title('The number of class in defects')

plt.bar(['defect', 'not defect'], counts, color='k')

plt.text(0, counts[0]+500, counts[0])

plt.text(1, counts[1]+500, counts[1])
counts = np.empty(4)

for i in range(4):

    counts[i] = np.sum(df_rle.iloc[np.arange(len(df_rle))%4==i, 1].notnull())

    plt.text(i, counts[i]+10, int(counts[i]))

plt.title('Defects on each class')

plt.bar(['class1', 'class2', 'class3', 'class4'], counts, color='k')
def mask2rle(img, width, height):

    rle = []

    lastColor = 0;

    currentPixel = 1;

    runStart = -1;

    runLength = 0;



    for x in range(width):

        for y in range(height):

            currentColor = img[y][x]

            if currentColor != lastColor:

                if currentColor == 255:

                    runStart = currentPixel

                    runLength = 1

                else:

                    rle.append(str(runStart))

                    rle.append(str(runLength))

                    runStart = -1

                    runLength = 0                    

            elif runStart > -1:

                runLength += 1

            lastColor = currentColor

            currentPixel += 1



    return " ".join(rle)



def rle2mask(rle, width, height):

    mask= np.zeros(width* height)

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for start, length in zip(starts, lengths):

        mask[start-1:start-1+length] = 255



    return mask.reshape(width, height).T
# return class-wise masks indicates where the defects are.

def get_mask(fname, width, height):

    masks = np.zeros((4, height, width))

    id_img = fname.split('/')[-1]

    for i in range(1, 5):

        classId_img = id_img + '_' + str(i)

        rle = df_rle[classId_img == df_rle['ImageId_ClassId']].iloc[0, 1]

        if type(rle)==str:

            masks[i-1] = rle2mask(rle, width, height)

    return masks
fig, axes = plt.subplots(8, 3, figsize=(15, 10))

axes[0, 0].set_ylabel('class1')

axes[1, 0].set_ylabel('class2')

axes[2, 0].set_ylabel('class3')

axes[3, 0].set_ylabel('class4')

for i in range(2):

    for j in range(3):

        fname = file_train[np.random.randint(LEN_TRAIN)]

        img = np.asarray(Image.open(fname).convert('L'))

        masks = get_mask(fname, W, H)

        for k, mask in enumerate(masks):

            axes[i*4+k, j].imshow(img)

            y, x = np.argwhere(mask>0).T

            axes[i*4+k, j].scatter(x, y, alpha=0.1, c='r', s=0.01)
def resize_img(img, resize_w, resize_h):

    img = Image.fromarray(img)

    w, h = img.size

    re_img = img.resize((resize_w, resize_h))

    return np.asarray(re_img)



def im2NHWC(img):

    ret = resize_img(img, resize_w, resize_h)

    ret = np.expand_dims(ret, axis=0)

    ret = np.expand_dims(ret, axis=3)

    return ret



def NHWC2im(nhwc):

    ret = np.squeeze(nhwc)

    ret = resize_img(ret, W, H)

    return ret
class training_generator(Sequence):

    def __init__(self, fnames, size_batch, w, h, idx_class, resize_w=W, resize_h=H):

        self.fnames = fnames

        self.size_batch = size_batch

        self.w = w

        self.h = h

        self.idx_class = idx_class

        self.resize_w = resize_w

        self.resize_h = resize_h

        self.on_epoch_end()

        

    def __load__(self, fname):

        img = np.asarray(Image.open(fname).convert('L'))

        mask = get_mask(fname, self.w, self.h)[self.idx_class]

        return img, mask    

     

    def __getitem__(self, idx_batch):

        if (idx_batch+1)*self.size_batch > len(self.fnames):

            self.size_batch = len(self.fnames) - idx_batch*self.size_batch

            

        fnames_batch = self.fnames[idx_batch*self.size_batch:(idx_batch+1)*self.size_batch]

        images = list()

        masks = list()

        

        for fname in fnames_batch:

            img, mask = self.__load__(fname)

            #img, mask = self.transform_item(img, mask)

            img = resize_img(img, self.resize_w, self.resize_h)

            mask = resize_img(mask, self.resize_w, self.resize_h)

            images.append(img)

            masks.append(mask)

        images = np.expand_dims(np.array(images), axis=3)

        masks = np.expand_dims(np.array(masks), axis=3)

        

        return images/255., masks/255.

    

    def on_epoch_end(self):

        pass

    

    def __len__(self):

        return int(np.ceil(len(self.fnames)/float(self.size_batch)))
# https://www.kaggle.com/cpmpml/fast-iou-metric-in-numpy-and-tensorflow

def get_iou_vector(A, B):

    # Numpy version

    

    batch_size = A.shape[0]

    metric = 0.0

    for batch in range(batch_size):

        t, p = A[batch], B[batch]

        true = np.sum(t)

        pred = np.sum(p)

        

        # deal with empty mask first

        if true == 0:

            metric += (pred == 0)

            continue

        

        intersection = np.sum(t * p)

        union = true + pred - intersection

        iou = intersection / union

        

        iou = np.floor(max(0, (iou - 0.45)*20)) / 10

        

        metric += iou

        

    metric /= batch_size

    return metric



def iou_metric(label, pred):

    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)
def dice_loss(pred, y):

    pred_f = K.flatten(pred)

    y_f = K.flatten(y)

    intersection = K.sum(y_f * pred_f)

    union = K.sum(y_f + pred_f)

    score = 2. * (intersection+1e-5) / (union+1e-05)

    return 1. - score



def dice_coef(pred, y):

    pred_f = pred.flatten()

    y_f = y.flatten()

    intersection = np.sum(y_f * pred_f)

    union = np.sum(y_f + pred_f)

    return 2. * (intersection+1e-5) / (union+1e-05)



def bce_dice_loss(y_true, y_pred):

    return losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
def bn_act_block(x, with_act=True):

    layer = BatchNormalization()(x)

    if with_act:

        layer = Activation('relu')(layer)

    return layer



def conv_block(x, n_filters, kernel_size=3, padding='same', strides=1):

    layer = bn_act_block(x)

    layer = Conv2D(n_filters, kernel_size, padding=padding, strides=strides)(layer)

    return layer



def initial_block(x, n_filters, kernel_size=3, padding='same', strides=1):

    layer = Conv2D(n_filters, kernel_size, padding=padding, strides=strides)(x)

    layer = conv_block(layer, n_filters, kernel_size, padding, strides)

    shortcut = Conv2D(n_filters, kernel_size=1, padding=padding, strides=strides)(x)

    shortcut = bn_act_block(shortcut, False)

    ret = Add()([layer, shortcut])

    return ret



def residual_block(x, n_filters, kernel_size=3, padding='same', strides=1):

    residue = conv_block(x, n_filters, kernel_size, padding, strides)

    residue = conv_block(residue, n_filters, kernel_size, padding, strides=1)

    shortcut = Conv2D(n_filters, kernel_size=1, padding=padding, strides=strides)(x)

    shortcut = bn_act_block(shortcut, False)

    ret = Add()([residue, shortcut])

    return ret



def upsample_concat_block(x, to_concat):

    upsampled = UpSampling2D(size=(2, 2))(x)

    ret = Concatenate()([upsampled, to_concat])

    return ret
SIZE_BATCH = 16

resize_w = W//4

resize_h = H//4

n_filters = [8, 16, 32, 64, 128]
def Res_Unet(h, w, pre_weights=None):   

    # Encoder

    inputs = Input((h, w, 1)) 

    e1 = initial_block(inputs, n_filters[0], strides=1)

    e2 = residual_block(e1, n_filters[1], strides=2)

    e3 = residual_block(e2, n_filters[2], strides=2)

    e4 = residual_block(e3, n_filters[3], strides=2)

    e5 = residual_block(e4, n_filters[4], strides=2)

    

    # Bridge

    b1 = conv_block(e5, n_filters[3], strides=1)

    b2 = conv_block(b1, n_filters[3], strides=1)

    

    # Decoder

    u1 = upsample_concat_block(b2, e4)

    d1 = residual_block(u1, n_filters[4])

    u2 = upsample_concat_block(d1, e3)

    d2 = residual_block(u2, n_filters[3])

    u3 = upsample_concat_block(d2, e2)

    d3 = residual_block(u3, n_filters[2])

    u4 = upsample_concat_block(d3, e1)

    d4 = residual_block(u4, n_filters[1])

    

    outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(d4)

    

    model = Model(input=inputs, output=outputs)

    model.compile(optimizer=Adam(lr=1e-02), loss=bce_dice_loss, metrics=[iou_metric])

    

    if(pre_weights):

        model.load_weights(pre_weights)

        

    return model



models = list()

for i in range(4):

    # load pre-trained model for inference processing.

    models.append(Res_Unet(resize_h, resize_w, pre_weights=f'../input/mymodel/model_class{i+1}.h5'))

    # build initial model for training.

    #models.append(Res_Unet(resize_h, resize_w'))

models[0].summary()
# for training

"""

# training with validation

history = [None, None, None, None]

n_epochs = [4, 4, 30, 10]

for i in range(4):

    early_stopping = EarlyStopping(monitor='val_loss')

    splited_train, splited_valid = train_test_split(file_train, train_size=0.9)

    train_generator = training_generator(splited_train,

                                         size_batch=SIZE_BATCH,

                                         w=W,

                                         h=H,

                                         idx_class=i,

                                         resize_w=resize_w,

                                         resize_h=resize_h)

    valid_generator = training_generator(splited_valid,

                                         size_batch=SIZE_BATCH,

                                         w=W,

                                         h=H,

                                         idx_class=i,

                                         resize_w=resize_w,

                                         resize_h=resize_h)

    history[i] = models[i].fit_generator(train_generator,

                                         steps_per_epoch=len(splited_train)//SIZE_BATCH,

                                         epochs=n_epochs[i],

                                         shuffle=True,

                                         verbose=1,

                                         validation_data=valid_generator,

                                         validation_steps=len(splited_valid)//SIZE_BATCH)



    eval(f"models[{i}].save_weights('model_class{i+1}')")

    print(f'class{i+1} training done.')

"""
# plot training histories.

"""

fig, axes = plt.subplots(2, 2, figsize=(18, 10))

for i , ax in enumerate(axes.flatten()):

    ax_t = ax.twinx()

    ax.plot(history[i].history['iou_metric'], 'b', label='trainig_acc')

    ax.plot(history[i].history['val_iou_metric'], 'y', label='validation_acc')

    ax_t.plot(history[i].history['loss'], 'r', label='dice_loss')

    ax_t.plot(history[i].history['val_loss'], 'g', label='validation_loss')

    ax.set_xlabel('epochs')

    ax.set_ylabel('my_iou_metric')

    ax_t.set_ylabel('loss')

    ax.legend(loc='right')

    ax_t.legend(loc='center left')

plt.show()

"""
sample_idx = np.random.randint(0, len(file_train), 3)

fig, axes = plt.subplots(12, 3, figsize=(15, 15))

axes[0, 0].set_title('original image')

axes[0, 1].set_title('masked image')

axes[0, 2].set_title('predicted image')

axes[0, 0].set_ylabel('class1')

axes[1, 0].set_ylabel('class2')

axes[2, 0].set_ylabel('class3')

axes[3, 0].set_ylabel('class4')

for i in range(3):

    fname = file_train[sample_idx[i]]

    img = np.asarray(Image.open(fname).convert('L'))

    masks = get_mask(fname, W, H)

    

    for j, mask in enumerate(masks):

        # draw background images

        axes[i*4+j, 0].imshow(img)

        axes[i*4+j, 1].imshow(img)

        axes[i*4+j, 2].imshow(img)

        

        # draw target masks

        if mask is not None:

            y, x = np.argwhere(mask>0).T

            axes[i*4+j, 1].scatter(x, y, alpha=0.1, c='r', s=0.01)

            

        # draw predicted segments

        nhwc = im2NHWC(img)/255.

        predicted = models[j].predict_on_batch(nhwc)

        predicted = NHWC2im(predicted)

        y, x = np.argwhere(predicted > 0.9).T

        axes[i*4+j, 2].scatter(x, y, alpha=0.1, c='r', s=0.01)
submission = pd.DataFrame(columns=['ImageId_ClassId', 'EncodedPixels'])

i = 0

for fname in tqdm_notebook(file_test):

    img = np.asarray(Image.open(fname).convert('L'))/255.

    nhwc = im2NHWC(img)

    for j in range(4):

        classId_test = fname.split('/')[-1] + '_' + str(j+1)



        # predict the defects from each model

        predicted = models[j].predict_on_batch(nhwc)

        predicted = NHWC2im(predicted)



        # take pixels bigger than threshold-value as masks.

        th_predicted = (predicted>0.9).astype(int)*255

        rle_predicted = mask2rle(th_predicted, 1600, 256)

        submission.loc[i] = [classId_test, rle_predicted]

        i+=1

        

submission.to_csv('./submission.csv', index=False)

submission.head()