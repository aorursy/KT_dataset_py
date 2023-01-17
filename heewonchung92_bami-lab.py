import os

import numpy as np

import cv2

import scipy.io as sio

import time



import keras

from keras.models import Sequential, Model

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization

from keras.optimizers import Adam

from keras import backend as K

from keras.losses import binary_crossentropy



import pydicom

from keras.preprocessing.image import load_img, img_to_array

from time import time

from sklearn.metrics import confusion_matrix
def read_png_true(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    img_png = img.copy()

    img_png = np.array(img_png.copy(), dtype=np.float64) 

    img_png /= np.max(img_png)

    img_png[img_png > 0] = 1

    return img_png



def transform_to_hu(medical_image, image):

    hu_image = image * medical_image.RescaleSlope + medical_image.RescaleIntercept

    hu_image[hu_image < -1024] = -1024

    return hu_image



def window_image(image, window_center, window_width):

    window_image = image.copy()

    img_min = window_center - (window_width / 2)

    img_max = window_center + (window_width / 2)

    window_image[window_image < img_min] = img_min

    window_image[window_image > img_max] = img_max

    return window_image



def resize_normalize(img):

    img = np.array(img, dtype=np.float64)

    img -= np.min(img)

    img /= np.max(img)

    return img



def read_dicom(path, window_widht=400, window_level=60):

    img_medical = pydicom.dcmread(path)

    img_data = img_medical.pixel_array



    img_hu = transform_to_hu(img_medical, img_data)

    img_window = window_image(img_hu.copy(), window_level, window_widht)

    img_window_norm = resize_normalize(img_window)



    img_window_norm = np.expand_dims(img_window_norm, axis=2)   # (512, 512, 1)

    img_ths = np.concatenate([img_window_norm, img_window_norm, img_window_norm], axis=2)   # (512, 512, 3)

    return img_ths



def dice_coeff(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return score



def dice_loss(y_true, y_pred):

    loss = 1 - dice_coeff(y_true, y_pred)

    return loss



def bce_dice_loss(y_true, y_pred):

    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    return loss



def unet_bami(input_size, num_classes):

    inputs = Input(input_size)

    # 512

    down0a = Conv2D(16, (3, 3), padding='same')(inputs)

    down0a = BatchNormalization()(down0a)

    down0a = Activation('relu')(down0a)

    down0a = Conv2D(16, (3, 3), padding='same')(down0a)

    down0a = BatchNormalization()(down0a)

    down0a = Activation('relu')(down0a)

    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)

    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)

    down0 = BatchNormalization()(down0)

    down0 = Activation('relu')(down0)

    down0 = Conv2D(32, (3, 3), padding='same')(down0)

    down0 = BatchNormalization()(down0)

    down0 = Activation('relu')(down0)

    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)

    down1 = BatchNormalization()(down1)

    down1 = Activation('relu')(down1)

    down1 = Conv2D(64, (3, 3), padding='same')(down1)

    down1 = BatchNormalization()(down1)

    down1 = Activation('relu')(down1)

    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)

    down2 = BatchNormalization()(down2)

    down2 = Activation('relu')(down2)

    down2 = Conv2D(128, (3, 3), padding='same')(down2)

    down2 = BatchNormalization()(down2)

    down2 = Activation('relu')(down2)

    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)

    down3 = BatchNormalization()(down3)

    down3 = Activation('relu')(down3)

    down3 = Conv2D(256, (3, 3), padding='same')(down3)

    down3 = BatchNormalization()(down3)

    down3 = Activation('relu')(down3)

    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)

    down4 = BatchNormalization()(down4)

    down4 = Activation('relu')(down4)

    down4 = Conv2D(512, (3, 3), padding='same')(down4)

    down4 = BatchNormalization()(down4)

    down4 = Activation('relu')(down4)

    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)

    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)

    center = BatchNormalization()(center)

    center = Activation('relu')(center)

    center = Conv2D(1024, (3, 3), padding='same')(center)

    center = BatchNormalization()(center)

    center = Activation('relu')(center)

    # center

    up4 = UpSampling2D((2, 2))(center)

    up4 = concatenate([down4, up4], axis=3)

    up4 = Conv2D(512, (3, 3), padding='same')(up4)

    up4 = BatchNormalization()(up4)

    up4 = Activation('relu')(up4)

    up4 = Conv2D(512, (3, 3), padding='same')(up4)

    up4 = BatchNormalization()(up4)

    up4 = Activation('relu')(up4)

    up4 = Conv2D(512, (3, 3), padding='same')(up4)

    up4 = BatchNormalization()(up4)

    up4 = Activation('relu')(up4)

    # 16

    up3 = UpSampling2D((2, 2))(up4)

    up3 = concatenate([down3, up3], axis=3)

    up3 = Conv2D(256, (3, 3), padding='same')(up3)

    up3 = BatchNormalization()(up3)

    up3 = Activation('relu')(up3)

    up3 = Conv2D(256, (3, 3), padding='same')(up3)

    up3 = BatchNormalization()(up3)

    up3 = Activation('relu')(up3)

    up3 = Conv2D(256, (3, 3), padding='same')(up3)

    up3 = BatchNormalization()(up3)

    up3 = Activation('relu')(up3)

    # 32

    up2 = UpSampling2D((2, 2))(up3)

    up2 = concatenate([down2, up2], axis=3)

    up2 = Conv2D(128, (3, 3), padding='same')(up2)

    up2 = BatchNormalization()(up2)

    up2 = Activation('relu')(up2)

    up2 = Conv2D(128, (3, 3), padding='same')(up2)

    up2 = BatchNormalization()(up2)

    up2 = Activation('relu')(up2)

    up2 = Conv2D(128, (3, 3), padding='same')(up2)

    up2 = BatchNormalization()(up2)

    up2 = Activation('relu')(up2)

    # 64

    up1 = UpSampling2D((2, 2))(up2)

    up1 = concatenate([down1, up1], axis=3)

    up1 = Conv2D(64, (3, 3), padding='same')(up1)

    up1 = BatchNormalization()(up1)

    up1 = Activation('relu')(up1)

    up1 = Conv2D(64, (3, 3), padding='same')(up1)

    up1 = BatchNormalization()(up1)

    up1 = Activation('relu')(up1)

    up1 = Conv2D(64, (3, 3), padding='same')(up1)

    up1 = BatchNormalization()(up1)

    up1 = Activation('relu')(up1)

    # 128

    up0 = UpSampling2D((2, 2))(up1)

    up0 = concatenate([down0, up0], axis=3)

    up0 = Conv2D(32, (3, 3), padding='same')(up0)

    up0 = BatchNormalization()(up0)

    up0 = Activation('relu')(up0)

    up0 = Conv2D(32, (3, 3), padding='same')(up0)

    up0 = BatchNormalization()(up0)

    up0 = Activation('relu')(up0)

    up0 = Conv2D(32, (3, 3), padding='same')(up0)

    up0 = BatchNormalization()(up0)

    up0 = Activation('relu')(up0)

    # 256

    up0a = UpSampling2D((2, 2))(up0)

    up0a = concatenate([down0a, up0a], axis=3)

    up0a = Conv2D(16, (3, 3), padding='same')(up0a)

    up0a = BatchNormalization()(up0a)

    up0a = Activation('relu')(up0a)

    up0a = Conv2D(16, (3, 3), padding='same')(up0a)

    up0a = BatchNormalization()(up0a)

    up0a = Activation('relu')(up0a)

    up0a = Conv2D(16, (3, 3), padding='same')(up0a)

    up0a = BatchNormalization()(up0a)

    up0a = Activation('relu')(up0a)

    # 512

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0a)



    model = Model(inputs, classify)

    model.compile(optimizer=Adam(lr=5e-4), loss=bce_dice_loss, metrics=[dice_coeff])

    return model
basic_dir = '../input/urinary-stone-challenge/'

images_dir_valid_dcm = basic_dir + 'Valid/DCM/'

images_dir_valid_mask = basic_dir + 'Valid/Mask/'

images_dir_test_dcm = basic_dir + 'Test/DCM/'



img_size = (512, 512, 3)

img_resize = (512, 512)

valid_idx = np.arange(601, 700+1)

test_idx = np.arange(701, 900+1)



model = unet_bami(input_size=img_size, num_classes=1)

model.load_weights('../input/bami-model/model_weight_30.h5')

model.summary()
arr_IOU = []

ths_prob = 0.5

for idx in valid_idx:

    ### True

    input_true = read_png_true(images_dir_valid_mask + str(idx) + '.png')



    ### Predict

    img_dcm = read_dicom(images_dir_valid_dcm + str(idx) + '.dcm')

    img = np.expand_dims(img_dcm, axis=0)

    input_pred = model.predict(img)

    input_pred = input_pred.squeeze()



    input_pred[input_pred < ths_prob] = 0

    input_pred[input_pred >= ths_prob] = 1



    ### Confusions

    y_true = np.asarray(input_true.flatten(), dtype=np.int32)

    y_pred = np.asarray(input_pred.flatten(), dtype=np.int32)

    cfs_mtx = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # print('TN: ', cfs_mtx[0][0], ', FP: ', cfs_mtx[0][1], ', FN: ', cfs_mtx[1][0], ', TP: ', cfs_mtx[1][1])



    val_sum = cfs_mtx[0][1] + cfs_mtx[1][0] + cfs_mtx[1][1]

    val_tp = cfs_mtx[1][1]

    val_iou = np.asarray(val_tp, dtype=np.float64) / np.asarray(val_sum, dtype=np.float64)



    arr_IOU.append(val_iou)

    print('Num: ', idx, ', val_tp: ', val_tp, ', val_sum: ', val_sum, ', IOU: ', val_iou)

arr_IOU = np.asarray(arr_IOU)

print('Mean: ', np.mean(arr_IOU))
### pip install openpyxl

import pandas as pd

df_excel = pd.DataFrame({'val_name': valid_idx, 'IOU': arr_IOU})

df_excel.to_excel('./Valid_IOU.xlsx')



os.chdir('/kaggle/working')

from IPython.display import FileLink

FileLink('Valid_IOU.xlsx')
Save_img_folder_mask ='./Test_Result_Mask/'

if not os.path.exists(Save_img_folder_mask):

    os.makedirs(Save_img_folder_mask)

    

Save_img_folder_heatmap ='./Test_Result_Heatmap/'

if not os.path.exists(Save_img_folder_heatmap):

    os.makedirs(Save_img_folder_heatmap)

    

    

ths_prob = 0.5

for idx in test_idx:

    ### Predict

    img_dcm = read_dicom(images_dir_test_dcm + str(idx) + '.dcm')

    img = np.expand_dims(img_dcm, axis=0)

    input_pred = model.predict(img)

    input_pred = input_pred.squeeze()



    input_pred[input_pred < ths_prob] = 0

    input_pred[input_pred >= ths_prob] = 1



    img_dcm = np.asarray(img_dcm.copy(), np.float64) * 255

    img_pred = np.asarray(input_pred.copy(), np.float64) * 255

    cv2.imwrite(Save_img_folder_mask + 'test_pred_' + str(idx) + '.png', img_pred)



    mask_pred = img_dcm.copy()

    for v_row in range(0, 512):

        for v_col in range(0, 512):

            if input_pred[v_row, v_col] != 0:

                mask_pred[v_row, v_col, 0] = 0

                mask_pred[v_row, v_col, 1] = 0

    cv2.imwrite(Save_img_folder_heatmap + 'test_dcm_pred_' + str(idx) + '.png', mask_pred)

    print('Num: ', idx, img_dcm.shape, img_pred.shape, mask_pred.shape)