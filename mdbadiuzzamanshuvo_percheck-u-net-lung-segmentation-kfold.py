!mkdir ../input_c
!mkdir ../input_c/segmentation
!mkdir ../input_c/segmentation/test
!mkdir ../input_c/segmentation/train
!mkdir ../input_c/segmentation/train/augmentation
!mkdir ../input_c/segmentation/train/image
!mkdir ../input_c/segmentation/train/mask
!mkdir ../input_c/segmentation/train/dilate
!ls ../input_c/segmentation
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from glob import glob
from tqdm import tqdm
INPUT_C_DIR = os.path.join("..", "input_c")
INPUT_DIR = os.path.join("..", "input")

SEGMENTATION_DIR = os.path.join(INPUT_C_DIR, "segmentation")
SEGMENTATION_TEST_DIR = os.path.join(SEGMENTATION_DIR, "test")
SEGMENTATION_TRAIN_DIR = os.path.join(SEGMENTATION_DIR, "train")
SEGMENTATION_AUG_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "augmentation")
SEGMENTATION_IMAGE_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "image")
SEGMENTATION_MASK_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "mask")
SEGMENTATION_DILATE_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "dilate")
SEGMENTATION_SOURCE_DIR = os.path.join(INPUT_DIR, \
                                       "pulmonary-chest-xray-abnormalities")

SHENZHEN_TRAIN_DIR = os.path.join(SEGMENTATION_SOURCE_DIR, "ChinaSet_AllFiles", \
                                  "ChinaSet_AllFiles")
SHENZHEN_IMAGE_DIR = os.path.join(SHENZHEN_TRAIN_DIR, "CXR_png")
SHENZHEN_MASK_DIR = os.path.join(INPUT_DIR, "shcxr-lung-mask", "mask", "mask")

MONTGOMERY_TRAIN_DIR = os.path.join(SEGMENTATION_SOURCE_DIR, \
                                    "Montgomery", "MontgomerySet")
MONTGOMERY_IMAGE_DIR = os.path.join(MONTGOMERY_TRAIN_DIR, "CXR_png")
MONTGOMERY_LEFT_MASK_DIR = os.path.join(MONTGOMERY_TRAIN_DIR, \
                                        "ManualMask", "leftMask")
MONTGOMERY_RIGHT_MASK_DIR = os.path.join(MONTGOMERY_TRAIN_DIR, \
                                         "ManualMask", "rightMask")

DILATE_KERNEL = np.ones((15, 15), np.uint8)

BATCH_SIZE=2

#Prod
EPOCHS=35

#Desv
#EPOCHS=16
print(MONTGOMERY_LEFT_MASK_DIR)
!ls MONTGOMERY_LEFT_MASK_DIR
montgomery_left_mask_dir = glob(os.path.join(MONTGOMERY_LEFT_MASK_DIR, '*.png'))
#montgomery_test = montgomery_left_mask_dir[0:50]
montgomery_train= montgomery_left_mask_dir[:]

for left_image_file in tqdm(montgomery_left_mask_dir):
    base_file = os.path.basename(left_image_file)
    image_file = os.path.join(MONTGOMERY_IMAGE_DIR, base_file)
    right_image_file = os.path.join(MONTGOMERY_RIGHT_MASK_DIR, base_file)

    image = cv2.imread(image_file)
    left_mask = cv2.imread(left_image_file, cv2.IMREAD_GRAYSCALE)
    right_mask = cv2.imread(right_image_file, cv2.IMREAD_GRAYSCALE)
    
    image = cv2.resize(image, (512, 512))
    left_mask = cv2.resize(left_mask, (512, 512))
    right_mask = cv2.resize(right_mask, (512, 512))
    
    mask = np.maximum(left_mask, right_mask)
    mask_dilate = cv2.dilate(mask, DILATE_KERNEL, iterations=1)
    
    if (left_image_file in montgomery_train):
        cv2.imwrite(os.path.join(SEGMENTATION_IMAGE_DIR, base_file), \
                    image)
        cv2.imwrite(os.path.join(SEGMENTATION_MASK_DIR, base_file), \
                    mask)
        cv2.imwrite(os.path.join(SEGMENTATION_DILATE_DIR, base_file), \
                    mask_dilate)
def add_colored_dilate(image, mask_image, dilate_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    dilate_image_gray = cv2.cvtColor(dilate_image, cv2.COLOR_BGR2GRAY)
    
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)
    dilate = cv2.bitwise_and(dilate_image, dilate_image, mask=dilate_image_gray)
    
    mask_coord = np.where(mask!=[0,0,0])
    dilate_coord = np.where(dilate!=[0,0,0])

    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]
    dilate[dilate_coord[0],dilate_coord[1],:] = [0,0,255]

    ret = cv2.addWeighted(image, 0.7, dilate, 0.3, 0)
    ret = cv2.addWeighted(ret, 0.7, mask, 0.3, 0)

    return ret

def add_colored_mask(image, mask_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
                                        
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)
    
    mask_coord = np.where(mask!=[0,0,0])

    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]

    ret = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

    return ret

def diff_mask(ref_image, mask_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)
    
    mask_coord = np.where(mask!=[0,0,0])

    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]

    ret = cv2.addWeighted(ref_image, 0.7, mask, 0.3, 0)
    return ret
shenzhen_mask_dir = glob(os.path.join(SHENZHEN_MASK_DIR, '*.png'))
#shenzhen_test = shenzhen_mask_dir[0:50]
shenzhen_train= shenzhen_mask_dir[:]

for mask_file in tqdm(shenzhen_mask_dir):
    base_file = os.path.basename(mask_file).replace("_mask", "")
    image_file = os.path.join(SHENZHEN_IMAGE_DIR, base_file)

    image = cv2.imread(image_file)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        
    image = cv2.resize(image, (512, 512))
    mask = cv2.resize(mask, (512, 512))
    mask_dilate = cv2.dilate(mask, DILATE_KERNEL, iterations=1)
    
    if (mask_file in shenzhen_train):
        cv2.imwrite(os.path.join(SEGMENTATION_IMAGE_DIR, base_file), \
                    image)
        cv2.imwrite(os.path.join(SEGMENTATION_MASK_DIR, base_file), \
                    mask)
        cv2.imwrite(os.path.join(SEGMENTATION_DILATE_DIR, base_file), \
                    mask_dilate)
train_files = glob(os.path.join(SEGMENTATION_IMAGE_DIR, "*.png"))
#test_files = glob(os.path.join(SEGMENTATION_TEST_DIR, "*.png"))
mask_files = glob(os.path.join(SEGMENTATION_MASK_DIR, "*.png"))
dilate_files = glob(os.path.join(SEGMENTATION_DILATE_DIR, "*.png"))

(len(train_files), \
 len(mask_files), \
 len(dilate_files))
# From: https://github.com/zhixuhao/unet/blob/master/data.py
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
        x_col = "dilate",
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
# From: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.square(y_true)) + K.sum(K.square(y_pred))
    jac = (intersection) / (sum_ - intersection)
    return jac
def unet(input_size=(256,256,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])
import pandas
from sklearn.model_selection import KFold

df = pandas.DataFrame(data={"filename": train_files, 'mask' : mask_files, 'dilate' : dilate_files})

kf = KFold(n_splits = 5, shuffle=False)
!wget https://www.kaggleusercontent.com/kf/37294602/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..fH1SXO36UtwK4_1dHA_zNQ.U1KCfXSyahnKO6JXCaetGblqUqlnuPS9tnmlsnitW0BQV9ExJYWb-tqN84tdq_psoPR0367fy2-ZWb-k_sRmIU42oGAG3bxRxdf6CUITJ5-jPCHGPncwdO7PThiw7D-m6VtoqiHfzFsZDPbD75uWagzStfu5Wv5kzmqmIfIdcgQ_cihrIeFkHluFoFF1BcEncSr9AQMNPTqypPbbG78B26CYOE688MqW_8naKlxMgWhnIJExxj930SRfKaqP21FPWwOq8pi0SesCQv4z84uGJH-200w0ENJuvCuTxRoenuabSO5Sxk0pKILtH4ovkZfkvGtTZB9xTaT2TcvnzAhKOTeV9KFx-a_MgWCJXq3mm_0ydSyzn4eKcZwl10j1f2nKWGzsiWziTnWy44bezsUy90B6JuSFPUpUw9w_AwuuwfETR0yRYJX_HjCRbDoclpvw4522LrSfRDK0Gcs2NKFO0yHC6irknvYPiibz52VGIKBtLMMi0H_QQq36s30nxeiVFMybq_PlDDKawZMGIGqi0-U0JRetbdOKVM4EnjaxF5PmmYUFC6RBDw9hS-4e3O2gbdK856Nj1S464NqmlCQiBFAeoxGwyNkvhtsBKlPg0271SFyfWv89olsFHqzKUE9W3nmJ0jJAgAfXLYhLTzejuDftLmcoVifRtg1RNhR-x6PAabQNHNaX0KlrBaShtIle.oLdmeCkE_8--mepqvEaajA/1_unet_lung_seg.hdf5
!wget https://www.kaggleusercontent.com/kf/37601327/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..RddFEITkTtfkQ__i7yrHgQ.tv18M7f1NqHVQog_TlJEo26Smmy-EnRXLMeoCJIWBrS4K2ZsVxE8l29Uc8ESC02M2gPe-6OVwgxNiK0MLLJcFNpMBgu85dJRgsGBM0GgAKNMVPrvB2viIFn3LdNsT1vcms2JJx0E2ZQ_POFU37eMET8AEwJaeFbwk_xSX4yLaUWeeim3wEo0AGehCk_rZYUSBFoENrhR0psqoaIaCxF5oL5_goBoqJfHcWsCRM_mLAXi30UUD0IFOByCabCJ0_GCgQHpqcuS-6LufbAenRgCy6-wT-nhg9XPYI2hD-npzezysMjapLND_Nsv7yaz6Kl6trvYe-WK6hd55XIfLWuV90sFk6h9lxUL15QhYDUtRM3DdphoJ9rmIFe8jZtzr7RKyY6y-U8p5B8kfeTFWIFRqeOLXupUeWOwe2gG6egceiQgl4-Yz9bruKCmcqPcQzRHmSU1c5nv9aIDCNEzJ1jqHJflSUoIIaYSqGOIyF5kJvWqpFWaByACbcreh5NQ5FRWvxAhvBBz-5A8TOKihq3dgVFkppjDbfh7nloq3_BnFlaKBIn3BOZORlnj48BOS9XcZj95tY0wAmbtcSRaTliVLu8xnsITeARsk4jONTm3_eAN9fR8vAKr8L3HHo06PMY94uwBERDAMHyFAchCVuBGuQ1GjrWInQJAf8GxxUYrDTKvUUUwCxUq3-j2ElMEOf_B.lw7NGvJxR7gnQPu_zpIotw/1_unet_lung_seg.hdf5
!wget https://www.kaggleusercontent.com/kf/37672840/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..__qFt9DeBUTsJudb3q26Jg.jwIHafkgbCucmcdzJ4EOrMm17QTcX5o-j07s7E5bdZUeI8a9vJZr69nA7nyguHvjfV70Q96JX55FQrPdGlCKl-Ju0_KXCvfxRFgsMgZ2E2nv9ifSSs9Nf0a_Xc-MC7OGaptbT0qNUFh4egG8rRBDv2czv0cFSj7B5AdTp1vK_mV6xo6ZTlmdOdBs91jp_7Za_HQsEiwyZzm6g2ibeoQAvGs-nS1ysZTLAjU7RUzbaMXG4KLr-XiXZ0VQWEioALqcSm43meAZhPAFNLufyepWHbA0TuKL_UIbzRwLzPqutn7R0Z-eVQkshSiL1QXfGJdDJzjE4Fq2Jm68h7WGjQ0AOUjqFSeQJVS7tpzeHt19d3J6rL4BVcrdg9wBxhpPxFp-1oZ-hi2oNFbAwbY01XZULEuyKLF_1EwV9m61xYeEMLgz44BmQjsE2YhEcc3t9rxvrgYBqxMp6ya1IxNrz2CTLjH_c8JOZfoIsyde7mOSu_1tJNZsGc9g27AXtGUepPztZXhWqrK0B6VQN-cQgMjibqtNsm1j84NzucQ2rvLYzPSxuXAULCaT8_gkfo4nWRAhD8vndiJ-_Zbj59erLiNvzWbRladvfW4Rxrk2wGDeK5bIL-sBXvb4NQd7uf7jZAfFPWqqKEApr870BpHjwzNVKX12qI5zCotuhA94dbqbhugdj70uA4jxGccvgNGwNAZJ.v9HxRv2O7gu_A1lIoMgdBA/1_unet_lung_seg.hdf5
model_unet = load_model('1_unet_lung_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})
model_trans = load_model('1_unet_lung_seg.hdf5.1', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})
model_segnet = load_model('1_unet_lung_seg.hdf5.2', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})
start = 0
span = 10

if start+span > len(train_files):
    start = 0

for i in range(start, start+span):
    img = cv2.imread(train_files[i],cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512,512))
    img =img/255
    img=img[np.newaxis, :, :, np.newaxis]
    
    pred_unet = model_unet.predict(img)
    pred_trans = model_trans.predict(img)
    pred_segnet = model_segnet.predict(img)

    plt.figure(figsize=(12,12))
    plt.subplot(1,5,1)
    plt.imshow(cv2.imread(train_files[i]))
    plt.title('Original Image')
    plt.subplot(1,5,2)
    plt.imshow(np.squeeze(cv2.imread(dilate_files[i])))
    plt.title('Original Mask')
    
    plt.subplot(1,5,3)
    plt.imshow(np.squeeze(pred_segnet) > .5)
    plt.title('SegNet')
    plt.subplot(1,5,4)
    plt.imshow(np.squeeze(pred_unet) > .5)
    plt.title('U-Net')
    plt.subplot(1,5,5)
    plt.imshow(np.squeeze(pred_trans) > .5)
    plt.title('TransResU-Net')
    
    plt.show()
