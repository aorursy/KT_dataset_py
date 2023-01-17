import os

import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import cv2

from tqdm import tqdm

from glob import glob

from PIL import Image

from skimage.transform import resize

from sklearn.model_selection import train_test_split, KFold



import tensorflow as tf

import tensorflow.keras

from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



K.set_image_data_format('channels_last')
path = "../input/ultrasound-nerve-segmentation/train/"

file_list = os.listdir(path)

file_list[:20]
train_image = []

train_mask = glob(path + '*_mask*')



for i in train_mask:

    train_image.append(i.replace('_mask', ''))

        

print(train_image[:10],"\n" ,train_mask[:10])
# Display the first image and mask of the first subject.

image1 = np.array(Image.open(path+"1_1.tif"))

image1_mask = np.array(Image.open(path+"1_1_mask.tif"))

image1_mask = np.ma.masked_where(image1_mask == 0, image1_mask)



fig, ax = plt.subplots(1,3,figsize = (16,12))

ax[0].imshow(image1, cmap = 'gray')



ax[1].imshow(image1_mask, cmap = 'gray')



ax[2].imshow(image1, cmap = 'gray', interpolation = 'none')

ax[2].imshow(image1_mask, cmap = 'jet', interpolation = 'none', alpha = 0.7)
width = 128

height = 128
from tensorflow.keras.models import Model, load_model

from tensorflow.keras import Input

from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate,add

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
def dice_coef(y_true, y_pred):

    smooth = 0.0

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def jacard(y_true, y_pred):



    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum ( y_true_f * y_pred_f)

    union = K.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)



    return intersection/union





def dice_coef_loss(y_true, y_pred):

    return -dice_coef(y_true, y_pred)



# def iou(y_true, y_pred):

#     intersection = K.sum(y_true * y_pred)

#     sum_ = K.sum(y_true) + K.sum(y_pred)

#     jac = (intersection + smooth) / (sum_ - intersection + smooth)

#     return jac
def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):

    '''

    2D Convolutional layers

    

    Arguments:

        x {keras layer} -- input layer 

        filters {int} -- number of filters

        num_row {int} -- number of rows in filters

        num_col {int} -- number of columns in filters

    

    Keyword Arguments:

        padding {str} -- mode of padding (default: {'same'})

        strides {tuple} -- stride of convolution operation (default: {(1, 1)})

        activation {str} -- activation function (default: {'relu'})

        name {str} -- name of the layer (default: {None})

    

    Returns:

        [keras layer] -- [output layer]

    '''



    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)

    x = BatchNormalization(axis=3, scale=False)(x)



    if(activation == None):

        return x



    x = Activation(activation, name=name)(x)



    return x





def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):

    '''

    2D Transposed Convolutional layers

    

    Arguments:

        x {keras layer} -- input layer 

        filters {int} -- number of filters

        num_row {int} -- number of rows in filters

        num_col {int} -- number of columns in filters

    

    Keyword Arguments:

        padding {str} -- mode of padding (default: {'same'})

        strides {tuple} -- stride of convolution operation (default: {(2, 2)})

        name {str} -- name of the layer (default: {None})

    

    Returns:

        [keras layer] -- [output layer]

    '''



    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)

    x = BatchNormalization(axis=3, scale=False)(x)

    

    return x





def MultiResBlock(U, inp, alpha = 1.67):

    '''

    MultiRes Block

    

    Arguments:

        U {int} -- Number of filters in a corrsponding UNet stage

        inp {keras layer} -- input layer 

    

    Returns:

        [keras layer] -- [output layer]

    '''



    W = alpha * U



    shortcut = inp



    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +

                         int(W*0.5), 1, 1, activation=None, padding='same')



    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,

                        activation='relu', padding='same')



    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,

                        activation='relu', padding='same')



    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,

                        activation='relu', padding='same')



    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)

    out = BatchNormalization(axis=3)(out)



    out = add([shortcut, out])

    out = Activation('relu')(out)

    out = BatchNormalization(axis=3)(out)



    return out





def ResPath(filters, length, inp):

    '''

    ResPath

    

    Arguments:

        filters {int} -- [description]

        length {int} -- length of ResPath

        inp {keras layer} -- input layer 

    

    Returns:

        [keras layer] -- [output layer]

    '''





    shortcut = inp

    shortcut = conv2d_bn(shortcut, filters, 1, 1,

                         activation=None, padding='same')



    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')



    out = add([shortcut, out])

    out = Activation('relu')(out)

    out = BatchNormalization(axis=3)(out)



    for i in range(length-1):



        shortcut = out

        shortcut = conv2d_bn(shortcut, filters, 1, 1,

                             activation=None, padding='same')



        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')



        out = add([shortcut, out])

        out = Activation('relu')(out)

        out = BatchNormalization(axis=3)(out)



    return out





def MultiResUnet(input_size=(256,256,1)):

    '''

    MultiResUNet

    

    Arguments:

        height {int} -- height of image 

        width {int} -- width of image 

        n_channels {int} -- number of channels in image

    

    Returns:

        [keras model] -- MultiResUNet model

    '''





    inputs = Input(input_size)



    mresblock1 = MultiResBlock(32, inputs)

    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)

    mresblock1 = ResPath(32, 4, mresblock1)



    mresblock2 = MultiResBlock(32*2, pool1)

    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)

    mresblock2 = ResPath(32*2, 3, mresblock2)



    mresblock3 = MultiResBlock(32*4, pool2)

    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)

    mresblock3 = ResPath(32*4, 2, mresblock3)



    mresblock4 = MultiResBlock(32*8, pool3)

    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)

    mresblock4 = ResPath(32*8, 1, mresblock4)



    mresblock5 = MultiResBlock(32*16, pool4)



    up6 = concatenate([Conv2DTranspose(

        32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)

    mresblock6 = MultiResBlock(32*8, up6)



    up7 = concatenate([Conv2DTranspose(

        32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)

    mresblock7 = MultiResBlock(32*4, up7)



    up8 = concatenate([Conv2DTranspose(

        32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)

    mresblock8 = MultiResBlock(32*2, up8)



    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(

        2, 2), padding='same')(mresblock8), mresblock1], axis=3)

    mresblock9 = MultiResBlock(32, up9)



    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')

    

    model = Model(inputs=[inputs], outputs=[conv10])



    return model
def train_generator(data_frame, batch_size, train_path, aug_dict,

        image_color_mode="grayscale",

        mask_color_mode="grayscale",

        image_save_prefix="image",

        mask_save_prefix="mask",

        save_to_dir=None,

        target_size=(256,256),

        seed=1):

    '''

    can generate image and mask at the same time use the same seed for

    image_datagen and mask_datagen to ensure the transformation for image

    and mask is the same if you want to visualize the results of generator,

    set save_to_dir = "your path"

    '''

    image_datagen = ImageDataGenerator(**aug_dict)

    mask_datagen = ImageDataGenerator(**aug_dict)

    

    image_generator = image_datagen.flow_from_dataframe(

        data_frame,

        directory = train_path,

        x_col = "filename",

        class_mode = None,

        color_mode = image_color_mode,

        target_size = target_size,

        batch_size = batch_size,

        save_to_dir = save_to_dir,

        save_prefix  = image_save_prefix,

        seed = seed)



    mask_generator = mask_datagen.flow_from_dataframe(

        data_frame,

        directory = train_path,

        x_col = "mask",

        class_mode = None,

        color_mode = mask_color_mode,

        target_size = target_size,

        batch_size = batch_size,

        save_to_dir = save_to_dir,

        save_prefix  = mask_save_prefix,

        seed = seed)



    train_gen = zip(image_generator, mask_generator)

    

    for (img, mask) in train_gen:

        img, mask = adjust_data(img, mask)

        yield (img,mask)



def adjust_data(img,mask):

    img = img / 255

    mask = mask / 255

    mask[mask > 0.5] = 1

    mask[mask <= 0.5] = 0

    

    return (img, mask)
pos_mask = []

pos_img = []

neg_mask = []

neg_img = []



for mask_path, img_path in zip(train_mask, train_image):

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if np.sum(mask) == 0:

        neg_mask.append(mask_path)

        neg_img.append(img_path)

    else:

        pos_mask.append(mask_path)

        pos_img.append(img_path)
!mkdir generated

!mkdir generated/img
def flip_up_down(img):

    newImg = img.copy()

    return cv2.flip(newImg, 0)



def flip_right_left(img):

    newImg = img.copy()

    return cv2.flip(newImg, 1)
gen_img = []

gen_mask = []



for (img_path, mask_path) in tqdm(zip(pos_img, pos_mask)):

    image_name = img_path.split('/')[-1].split('.')[0]



    uf_img_path = 'generated/img/'+image_name+'_uf.jpg'

    uf_mask_path = 'generated/img/'+image_name+'_uf_mask.jpg'

    rf_img_path = 'generated/img/'+image_name+'_rf.jpg'

    rf_mask_path = 'generated/img/'+image_name+'_rf_mask.jpg'



    img = cv2.imread(img_path)

    mask = cv2.imread(mask_path)



    uf_img = flip_up_down(img)

    uf_mask = flip_up_down(mask)

    rf_img = flip_right_left(img)

    rf_mask = flip_right_left(mask)



    cv2.imwrite(uf_img_path, uf_img)

    cv2.imwrite(uf_mask_path, uf_mask)

    cv2.imwrite(rf_img_path, rf_img)

    cv2.imwrite(rf_mask_path, rf_mask)

    

    gen_img.append(uf_img_path)

    gen_mask.append(uf_mask_path)

    gen_img.append(rf_img_path)

    gen_mask.append(rf_mask_path)
aug_img = gen_img + train_image

aug_mask = gen_mask + train_mask



df_ = pd.DataFrame(data={"filename": aug_img, 'mask' : aug_mask})

df = df_.sample(frac=1).reset_index(drop=True)



kf = KFold(n_splits = 5, shuffle=False)
histories = []

losses = []

accuracies = []

dicecoefs = []

jacards = []



train_generator_args = dict(rotation_range=0.2,

                            width_shift_range=0.05,

                            height_shift_range=0.05,

                            shear_range=0.05,

                            zoom_range=0.05,

                            horizontal_flip=True,

                            fill_mode='nearest')



EPOCHS = 50

BATCH_SIZE = 32



for k, (train_index, test_index) in enumerate(kf.split(df)):

    train_data_frame = df.iloc[train_index]

    test_data_frame = df.iloc[test_index]

    

    train_gen = train_generator(train_data_frame, BATCH_SIZE,

                                None,

                                train_generator_args,

                                target_size=(height, width))



    test_gener = train_generator(test_data_frame, BATCH_SIZE,

                                None,

                                dict(),

                                target_size=(height, width))



    model = MultiResUnet(input_size=(height,width, 1))

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, \

                      metrics=[jacard, dice_coef, 'binary_accuracy'])



    model_checkpoint = ModelCheckpoint(str(k+1) + '_unet_ner_seg.hdf5', 

                                       verbose=1, 

                                       save_best_only=True)



    history = model.fit_generator(train_gen,

                                  steps_per_epoch=len(train_data_frame) // BATCH_SIZE, 

                                  epochs=EPOCHS, 

                                  callbacks=[model_checkpoint],

                                  validation_data = test_gener,

                                  validation_steps=len(test_data_frame) // BATCH_SIZE)

    

    model = load_model(str(k+1) + '_unet_ner_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'jacard': jacard, 'dice_coef': dice_coef})

    

    test_gen = train_generator(test_data_frame, BATCH_SIZE,

                                None,

                                dict(),

                                target_size=(height, width))

    results = model.evaluate_generator(test_gen, steps=len(test_data_frame) // BATCH_SIZE)

    results = dict(zip(model.metrics_names,results))

    

    histories.append(history)

    accuracies.append(results['binary_accuracy'])

    losses.append(results['loss'])

    dicecoefs.append(results['dice_coef'])

    jacards.append(results['jacard'])

    

    break
for h, history in enumerate(histories):



    keys = history.history.keys()

    fig, axs = plt.subplots(1, len(keys)//2, figsize = (25, 5))

    fig.suptitle('No. ' + str(h+1) + ' Fold Results', fontsize=30)



    for k, key in enumerate(list(keys)[:len(keys)//2]):

        training = history.history[key]

        validation = history.history['val_' + key]



        epoch_count = range(1, len(training) + 1)



        axs[k].plot(epoch_count, training, 'r--')

        axs[k].plot(epoch_count, validation, 'b-')

        axs[k].legend(['Training ' + key, 'Validation ' + key])
print('average accuracy : ', np.mean(np.array(accuracies)), '+-', np.std(np.array(accuracies)))

print('average loss : ', np.mean(np.array(losses)), '+-', np.std(np.array(losses)))

print('average jacard : ', np.mean(np.array(jacards)), '+-', np.std(np.array(jacards)))

print('average dice_coe : ', np.mean(np.array(dicecoefs)), '+-', np.std(np.array(dicecoefs)))
model = load_model('1_unet_ner_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'jacard': jacard, 'dice_coef': dice_coef})
for i in range(20):

    index=np.random.randint(0,len(test_data_frame.index))

    print(i+1, index)

    img = cv2.imread(test_data_frame['filename'].iloc[index], cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (height, width))

    img = img[np.newaxis, :, :, np.newaxis]

    img = img / 255

    pred = model.predict(img)



    plt.figure(figsize=(12,12))

    plt.subplot(1,3,1)

    plt.imshow(np.squeeze(img))

    plt.title('Original Image')

    plt.subplot(1,3,2)

    plt.imshow(np.squeeze(cv2.resize(cv2.imread(test_data_frame['mask'].iloc[index]), (height, width))))

    plt.title('Original Mask')

    plt.subplot(1,3,3)

    plt.imshow(np.squeeze(pred) > .5)

    plt.title('Prediction')

    plt.show()
!rm -r generated                                                                          