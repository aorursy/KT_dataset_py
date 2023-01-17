# Deformable convolutions https://arxiv.org/pdf/1703.06211.pdf & https://github.com/felixlaumon/deform-conv
# Please, download the https://github.com/felixlaumon/deform-conv project and add the /deform-conv folder to you project files.
import time

def log(text):
    time_int = time.time() - start_time
    print('{}:{:02.0f} {}.'.format(int(time_int/60), time_int%60, text))
import cv2
import os
import numpy as np
import pandas as pd


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
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dropout, Lambda, Dense
from deform_conv.layers import ConvOffset2D


def conv_block(input, num_filters, filter_size=(3,3), dropout=0.3, kernel_initializer='he_normal', deform=True):
    #c1 = BatchNormalization()(input)
    c1 = Conv2D(num_filters, filter_size, activation='elu', padding='same', kernel_initializer=kernel_initializer) (input)
    c1 = BatchNormalization()(c1)
    #c1 = Dropout(dropout)(c1)
    c1 = ConvOffset2D(num_filters)(c1)
    c2 = Conv2D(num_filters, filter_size, strides=(2,2), activation='elu', padding='same', 
                kernel_initializer=kernel_initializer) (c1)  if deform else c1
    return c2

def conv_pool_block(input, num_filters, filter_size=(3,3), pool_size=(2,2), dropout=0.3):
    conv = conv_block(input, num_filters, filter_size=filter_size)
#     pool = MaxPooling2D(pool_size) (conv)
#     return conv, pool
    return conv, conv

def transp_conv_block(input, input_res, num_filters, filter_size=(3,3), transp_size=(2,2),
                      dropout=0.3, kernel_initializer='he_normal', deform=False):
    transp = Conv2DTranspose(num_filters, transp_size, strides=transp_size, padding='same',
                             kernel_initializer=kernel_initializer) (input)
    cnc = concatenate([transp, input_res])
    cb = conv_block(cnc, num_filters, filter_size=filter_size, deform=deform)
    return cb

def Unet(img_size):
    inputs = Input((img_size, img_size, 3))
    s = Lambda(lambda x: x / 255)(inputs)

    c1, cp1 = conv_pool_block(s, 8, dropout=0.1)
    c2, cp2 = conv_pool_block(cp1, 16, dropout=0.1)
    c3, cp3 = conv_pool_block(cp2, 32, dropout=0.1)
    c4, cp4 = conv_pool_block(cp3, 64, dropout=0.2)
    c5, cp5 = conv_pool_block(cp4, 128, dropout=0.2)
    c6, cp6 = conv_pool_block(cp5, 256, dropout=0.3)

    c_middle = conv_block(cp6, 512, dropout=0.3)

    tc6 = transp_conv_block(c_middle, c6, 256, dropout=0.3)
    tc5 = transp_conv_block(tc6, c5, 128, dropout=0.2)
    tc4 = transp_conv_block(tc5, c4, 64, dropout=0.2)
    tc3 = transp_conv_block(tc4, c3, 32, dropout=0.1)
    tc2 = transp_conv_block(tc3, c2, 16, dropout=0.1)
    tc1 = transp_conv_block(tc2, c1, 8, dropout=0.1)

    #outputs = Conv2D(1, (1, 1), activation='sigmoid') (tc1)
    tc1 = Conv2DTranspose(4, (2,2), strides=(2,2), padding='same',
                             kernel_initializer='he_normal') (tc1)
    outputs = Dense(1, activation='sigmoid') (tc1)

    model = Model(inputs=[inputs], outputs=[outputs])
    #model.summary()
    return model
from keras.preprocessing.image import ImageDataGenerator

def generator(xtr, xval, ytr, yval, batch_size):
    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1)
    val_gen_args = dict()

    def get_gen(data, args):
        gen = ImageDataGenerator(**args)
        gen.fit(data, seed=7)
        gen = gen.flow(data, batch_size=batch_size, seed=7)
        return gen
    
    train_generator = zip(get_gen(xtr, data_gen_args), get_gen(ytr, data_gen_args))
    val_generator = zip(get_gen(xval, val_gen_args), get_gen(yval, val_gen_args))
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

def prob_to_rles(x, cutoff=0.4): # 0.5
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
def add_empty_ids(sub, test_ids, new_test_ids): 
    empty_ids = list(set(test_ids) - set(new_test_ids))
    print('debug: images without masks:', len(empty_ids))
    if len(empty_ids) == 0:
        return sub
    add_df = pd.DataFrame(empty_ids,columns=['ImageId'])
    add_df['EncodedPixels'] = ''
    return pd.concat([sub,add_df])


def save_submission(file_name, preds, img_path, img_sizes):
    img_ids = next(os.walk(img_path))[1]
    
    preds_upsampled = []
    for i in range(len(preds)):
        preds_upsampled.append(cv2.resize(preds[i], (img_sizes[i][1], img_sizes[i][0])))
        
    new_img_ids = []
    rles = []
    for n, id_ in enumerate(img_ids):
        rle = list(prob_to_rles(preds_upsampled[n]))
        rles.extend(rle)
        new_img_ids.extend([id_] * len(rle))
    sub = pd.DataFrame()
    sub['ImageId'] = new_img_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub = add_empty_ids(sub, img_ids, new_img_ids)
    
    print('debug: len(img_ids), len(sub.ImageId.unique())', len(img_ids), len(sub.ImageId.unique()))
    sub.to_csv(file_name, index=False)


import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model


if __name__ == "__main__":
    start_time = time.time()
    img_size = 256
    batch_size = 8 # 32
    epochs = 100
    
    data_path = 'data/'    
    train_path = data_path +'stage1/train/'
    test_path = data_path +'stage2/test/'
    
    log('Process Started')
    X_train, Y_train, X_test, sizes_test = make_df(train_path, test_path, img_size)
    log('Read data')
    xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=0.1, random_state=7)
    train_generator, val_generator = generator(xtr, xval, ytr, yval, batch_size)
    log('Data prepared')
    
    model = Unet(img_size)
    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[mean_iou])
    
    log('Model Fitting started...')
    earlystopper = EarlyStopping(patience=25, verbose=1)
    checkpointer = ModelCheckpoint('model.v3.h5', verbose=1, save_best_only=True)
    model.fit_generator(train_generator, steps_per_epoch=len(xtr)/6, epochs=epochs,
                        validation_data=val_generator, validation_steps=len(xval)/batch_size,
  #                     callbacks=[earlystopper, checkpointer]
                       )
    
   # model = load_model('model.v3.h5', custom_objects={'mean_iou': mean_iou, 'bce_dice_loss': bce_dice_loss})
   #                    #,'ConvOffset2D': ConvOffset2D})
    log('  Model Fitting finished')
    
    preds_test = model.predict(X_test, verbose=1)
    log('Test Predicted')

    save_submission('submission.v3.csv', preds_test, test_path, sizes_test)    
    log('Sibmission Created')

model.save('model.v3.h5')
# a visual ad-hock verification of the submission.
df = pd.read_csv('submission.v3.csv', nrows=2000)
df.head()
