from os import path                      # os level path manipulation

import numpy as np                       # array goodnes

from pandas import DataFrame, read_csv   # excel for python

from matplotlib import pyplot as plt     # plotting library

from pandas import DataFrame, read_csv   # excel for python

from tqdm import tqdm, trange            # progress bars



%matplotlib inline

plt.rcParams['figure.figsize'] = (12, 8) # set plot size
from glob import glob                            # Unix style pathname pattern expansion

from PIL import Image                            # image loading / saving

from skimage.exposure import equalize_adapthist  # CLAHE

from sklearn.model_selection import train_test_split





target_shape = (96, 96)

basepath = '../input/train/'





def preprocess_image(img, target_shape, clahe=False, onehot=False, cval=0):

    img.thumbnail(target_shape, Image.NEAREST) # resize and keep aspect ratio

    im = np.array(img)                         # convert to numpy array

    if clahe:

        im = equalize_adapthist(im)



    if len(im.shape) == 3:

        im = np.argmax(im, axis=-1)

    

    # padding to rarget_shape

    padding = np.abs(np.array(im.shape) - target_shape)

    lpad = padding // 2

    rpad = padding - lpad

    im = np.pad(im, [(lpad[0], rpad[0]), (lpad[1], rpad[1])],

                'constant', constant_values=cval)



    if onehot:

        a = []

        for idx in onehot:

            a.append(im == idx)

        im = np.stack(a, axis=2)

    else:

        im = im.reshape(target_shape+(1,))

        

    return im.astype(np.float)





def load_data(basepath, target_shape=(256, 256), max_samples=-1):

    X, Y = [], []

    for fn in tqdm(glob(path.join(basepath, 'BBBC010_v1_images', '*.tif')), desc='reading files'):

        cn = path.split(fn)[-1]

        wellcolumn = cn.split('_')[6]

        itype = cn.split('_')[7]

        if itype != 'w2': continue

        x_img = Image.open(fn)

        y_img = Image.open(path.join(basepath, 'BBBC010_v1_foreground', '%s_binary.png' % wellcolumn))

        x = preprocess_image(x_img, target_shape, clahe=True)

        y = preprocess_image(y_img, target_shape, clahe=False, onehot=[3, 0], cval=3)

        X.append(x), Y.append(y)

    return np.array(X), np.array(Y)





def data_augmentation(x, y):

    _x, _y = [], []

    for i in trange(len(x), desc='data augmentation'):

        _x.append(x[i])

        _y.append(y[i])

        _x.append(x[i,::-1])

        _y.append(y[i,::-1])

        _x.append(x[i,::-1,::-1])

        _y.append(y[i,::-1,::-1])

        _x.append(x[i,:,::-1])

        _y.append(y[i,:,::-1])

    return np.array(_x), np.array(_y)



X, Y = load_data('../input', target_shape=target_shape)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

X_train, Y_train = data_augmentation(X_train, Y_train)
from ipywidgets import interactive, fixed



def _plot(idx, x_y_p):

    X, Y, P = x_y_p

    idx = int(idx)

    if type(P) != np.ndarray:

        fig, axs = plt.subplots(1,3)

    else:

        fig, axs = plt.subplots(1,4)

    

    x = X[idx][...,0]

    y = Y[idx][...,1]

    

    axs[0].set_title('image')

    axs[0].imshow(x, cmap='plasma', vmin=0, vmax=1)

    

    axs[1].set_title('ground truth')

    axs[1].imshow(y)

    

    axs[2].set_title('image & ground truth')

    y_masked = np.copy(y)

    y_masked[y_masked==0] = np.NaN

    axs[2].imshow(x, cmap='Greys')

    axs[2].imshow(y_masked, cmap='Reds', vmin=0, vmax=1, alpha=0.4)

    

    if type(P) == np.ndarray:

        axs[3].set_title('prediction')

        axs[3].imshow(P[idx])

    

    plt.show()



def interactive_plot(x, y, p=None):

    return interactive(_plot, idx=range(len(x)), x_y_p=fixed((x,y,p)))



interactive_plot(X_train, Y_train)
interactive_plot(X_val, Y_val)
import keras

from keras.layers import Input, Conv2D, concatenate, AveragePooling2D

from keras.layers import UpSampling2D, BatchNormalization



channels_per_level = [32, 64, 128, 256]

bridge_channels = channels_per_level.pop()

identities = []

input_tensor = Input(shape=target_shape + (1,))

net = input_tensor



# encoder

for channels in channels_per_level:

    net = BatchNormalization(momentum=0.9)(net)

    net = Conv2D(channels, 3, padding='same', activation='relu')(net)

    identities.append(net)

    net = AveragePooling2D(padding='same')(net)



# bridge

net = BatchNormalization(momentum=0.9)(net)

net = Conv2D(bridge_channels, 3, padding='same', activation='relu')(net)



# decoder

for channels in channels_per_level[::-1]:

    net = UpSampling2D()(net)

    net = concatenate([net, identities.pop()])

    net = BatchNormalization(momentum=0.9)(net)

    net = Conv2D(channels, 3, padding='same', activation='relu')(net)



# classification

n_classes = Y_train[0].shape[-1]

net = Conv2D(n_classes, 1, padding='same', activation='sigmoid')(net)
import keras.backend as K

from keras.models import Model



#https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py

def dice_coef(y_true, y_pred, smooth=1.0):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_coef_loss(y_true, y_pred):

    return -dice_coef(y_true, y_pred)





learning_rate = 1e-3

model = Model(input_tensor, net)

model.compile(loss=dice_coef_loss,

              metrics=[dice_coef],

              optimizer=keras.optimizers.Adam(lr=learning_rate))
model.fit(X_train, Y_train, 

          validation_data=[X_val, Y_val],

          epochs=1, batch_size=1)
predictions = []

for i in trange(len(X_val)):

    im = X_val[i]

    im = im.reshape((1,)+im.shape)

    prediction = model.predict(im)[0]

    prediction = np.argmax(prediction, axis=-1)

    predictions.append(prediction)

predictions = np.array(predictions)
interactive_plot(X_val, Y_val, predictions)