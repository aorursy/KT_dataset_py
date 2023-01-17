import cv2

import os

import numpy as np





def make_df(train_path, test_path, img_size):

    train_ids = next(os.walk(train_path))[1]

    test_ids = next(os.walk(test_path))[1]

    X_train = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.uint8)

    Y_train = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)

    for i, id_ in enumerate(train_ids):

        path = train_path + id_

        img = cv2.imread(path + '/images/' + id_ + '.png')

        img = cv2.resize(img, (img_size, img_size))

        X_train[i] = img

        mask = np.zeros((img_size, img_size, 1), dtype=np.bool)

        for mask_file in next(os.walk(path + '/masks/'))[2]:

            mask_ = cv2.imread(path + '/masks/' + mask_file, 0)

            mask_ = cv2.resize(mask_, (img_size, img_size))

            mask_ = mask_[:, :, np.newaxis]

            mask = np.maximum(mask, mask_)

        Y_train[i] = mask

    X_test = np.zeros((len(test_ids), img_size, img_size, 3), dtype=np.uint8)

    sizes_test = []

    for i, id_ in enumerate(test_ids):

        path = test_path + id_

        img = cv2.imread(path + '/images/' + id_ + '.png')

        sizes_test.append([img.shape[0], img.shape[1]])

        img = cv2.resize(img, (img_size, img_size))

        X_test[i] = img



    return X_train, Y_train, X_test, sizes_test
from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.models import Model

from keras.layers import Input

from keras.layers.core import Dropout, Lambda





def Unet(img_size):

    inputs = Input((img_size, img_size, 3))

    s = Lambda(lambda x: x / 255)(inputs)



    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)

    c1 = Dropout(0.1)(c1)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)

    p1 = MaxPooling2D((2, 2))(c1)



    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)

    c2 = Dropout(0.1)(c2)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)

    p2 = MaxPooling2D((2, 2))(c2)



    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)

    c3 = Dropout(0.2)(c3)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)

    p3 = MaxPooling2D((2, 2))(c3)



    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)

    c4 = Dropout(0.2)(c4)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)

    p4 = MaxPooling2D(pool_size=(2, 2))(c4)



    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)

    c5 = Dropout(0.3)(c5)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)



    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)

    u6 = concatenate([u6, c4])

    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)

    c6 = Dropout(0.2)(c6)

    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)



    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)

    u7 = concatenate([u7, c3])

    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)

    c7 = Dropout(0.2)(c7)

    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)



    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)

    u8 = concatenate([u8, c2])

    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)

    c8 = Dropout(0.1)(c8)

    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)



    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)

    u9 = concatenate([u9, c1], axis=3)

    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)

    c9 = Dropout(0.1)(c9)

    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)



    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)



    model = Model(inputs=[inputs], outputs=[outputs])



    return model
from keras.preprocessing.image import ImageDataGenerator





def generator(xtr, xval, ytr, yval, batch_size):

    data_gen_args = dict(horizontal_flip=True,

                         vertical_flip=True,

                         rotation_range=90.,

                         width_shift_range=0.1,

                         height_shift_range=0.1,

                         zoom_range=0.1)

    image_datagen = ImageDataGenerator(**data_gen_args)

    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_datagen.fit(xtr, seed=7)

    mask_datagen.fit(ytr, seed=7)

    image_generator = image_datagen.flow(xtr, batch_size=batch_size, seed=7)

    mask_generator = mask_datagen.flow(ytr, batch_size=batch_size, seed=7)

    train_generator = zip(image_generator, mask_generator)



    val_gen_args = dict()

    image_datagen_val = ImageDataGenerator(**val_gen_args)

    mask_datagen_val = ImageDataGenerator(**val_gen_args)

    image_datagen_val.fit(xval, seed=7)

    mask_datagen_val.fit(yval, seed=7)

    image_generator_val = image_datagen_val.flow(xval, batch_size=batch_size, seed=7)

    mask_generator_val = mask_datagen_val.flow(yval, batch_size=batch_size, seed=7)

    val_generator = zip(image_generator_val, mask_generator_val)



    return train_generator, val_generator
import tensorflow as tf

import numpy as np

from keras import backend as K

from keras.losses import binary_crossentropy





def mean_iou(y_true, y_pred):

    prec = []

    for t in np.arange(0.5, 1.0, 0.05):

        y_pred_ = tf.to_int32(y_pred > t)

        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)

        K.get_session().run(tf.local_variables_initializer())

        with tf.control_dependencies([up_opt]):

            score = tf.identity(score)

        prec.append(score)

    return K.mean(K.stack(prec))



def dice_coef(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def bce_dice_loss(y_true, y_pred):

    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
from skimage.morphology import label



def rle_encoding(x):

    dots = np.where(x.T.flatten() == 1)[0]

    run_lengths = []

    prev = -2

    for b in dots:

        if (b>prev+1): run_lengths.extend((b + 1, 0))

        run_lengths[-1] += 1

        prev = b

    return run_lengths



def prob_to_rles(x, cutoff=0.5):

    lab_img = label(x > cutoff)

    for i in range(1, lab_img.max() + 1):

        yield rle_encoding(lab_img == i)
import cv2

import os

import pandas as pd

from sklearn.model_selection import train_test_split





if __name__ == "__main__":

    img_size = 256

    batch_size = 32

    train_path = '../input/stage1_train/'

    test_path = '../input/stage1_test/'

    

    X_train, Y_train, X_test, sizes_test = make_df(train_path, test_path, img_size)

    xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=0.1, random_state=7)

    train_generator, val_generator = generator(xtr, xval, ytr, yval, batch_size)

    

    model = Unet(img_size)

    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[mean_iou])

    

    model.fit_generator(train_generator, steps_per_epoch=len(xtr)/6, epochs=50,

                        validation_data=val_generator, validation_steps=len(xval)/batch_size)

    

    preds_test = model.predict(X_test, verbose=1)



    preds_test_upsampled = []

    for i in range(len(preds_test)):

        preds_test_upsampled.append(cv2.resize(preds_test[i], 

                                           (sizes_test[i][1], sizes_test[i][0])))

        

    test_ids = next(os.walk(test_path))[1]

    new_test_ids = []

    rles = []

    for n, id_ in enumerate(test_ids):

        rle = list(prob_to_rles(preds_test_upsampled[n]))

        rles.extend(rle)

        new_test_ids.extend([id_] * len(rle))

    sub = pd.DataFrame()

    sub['ImageId'] = new_test_ids

    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

    sub.to_csv('sub.csv', index=False)