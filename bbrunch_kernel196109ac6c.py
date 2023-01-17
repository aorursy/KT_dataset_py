!pip install -U segmentation-models==0.2.1

!pip install albumentations
import os, sys, gc

import pandas as pd

import numpy  as np

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from multiprocessing import Pool, cpu_count

from cv2 import resize

from skimage.io import imread as skiImgRead

from skimage.morphology import label

from sklearn.model_selection import train_test_split

from segmentation_models           import Unet

from segmentation_models.backbones import get_preprocessing

from segmentation_models.utils     import set_trainable

from segmentation_models.losses    import bce_jaccard_loss, bce_dice_loss

from segmentation_models.metrics   import iou_score, f2_score

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, LearningRateScheduler
IMG_HW    = 768

ZOOM_HW   = 320

DATA_DIR  = '../input/airbus-ship-detection'

TRAIN_DIR = os.path.join(DATA_DIR, 'train_v2')

TEST_DIR  = os.path.join(DATA_DIR, 'test_v2')
def rle_decode(rle_mask):

    s = rle_mask.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(IMG_HW*IMG_HW, dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(IMG_HW,IMG_HW).T



def rle_encode(im):

    pixels = im.flatten(order = 'F')

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def count_pix_inpool(df_col):

    pool = Pool()

    res = pool.map( count_pix, df_col.items() )

    pool.close()

    pool.join()

    return res



def count_pix(row):

    v = row[1]

    if v is np.nan or type(v) != str: 

        return v

    else:

        return rle_decode(v).sum()
train_csv  = pd.read_csv( os.path.join( DATA_DIR, 'train_ship_segmentations_v2.csv') )
DROP_NO_SHIP_FRACTION = 0.8



balanced_train_csv = (

    train_csv

    .set_index('ImageId')

    .drop(

        train_csv.loc[

            train_csv.isna().any(axis=1),

            'ImageId'

        ].sample( frac = DROP_NO_SHIP_FRACTION )

    )

    .reset_index()

)
b_train_csv, b_valid_csv = train_test_split(balanced_train_csv['ImageId'], test_size = 0.2)



b_train_csv = balanced_train_csv.set_index('ImageId').loc[b_train_csv].reset_index()

b_valid_csv = balanced_train_csv.set_index('ImageId').loc[b_valid_csv].reset_index()
BACKBONE  = 'resnet34'

preprocess_input = get_preprocessing(BACKBONE)
from albumentations import (

    Compose, HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, Transpose,

    OneOf, ToFloat,

    RandomBrightness, RandomContrast, RandomGamma, CLAHE,

    GridDistortion, ElasticTransform, JpegCompression,

    RGBShift, GaussNoise, IAAAdditiveGaussianNoise, HueSaturationValue,

    Blur, MotionBlur, MedianBlur, RandomBrightnessContrast,

    GridDistortion, OpticalDistortion, RandomSizedCrop, CenterCrop

)



augmentor = Compose([

    OneOf([

        HorizontalFlip(),

        VerticalFlip(),

        RandomRotate90(),

        Transpose(),

    ], p=0.8), 

    ShiftScaleRotate(rotate_limit=20),

    OneOf([

        MotionBlur(blur_limit=3),

        MedianBlur(blur_limit=3),

        Blur(blur_limit=3),

    ], p=0.3),

    OneOf([

        RandomGamma(),

        RandomContrast(),

        RandomBrightness(),

        CLAHE(),

     ], p=0.3),

    OneOf([

        IAAAdditiveGaussianNoise(),

        HueSaturationValue(),

        GaussNoise(),

    ], p=0.2),

    OneOf([

        ElasticTransform(),

        OpticalDistortion(),

        GridDistortion(),

    ], p=0.3),

    RandomSizedCrop(min_max_height=(IMG_HW/2, IMG_HW), height=IMG_HW, width=IMG_HW, p=0.3),

    ToFloat(max_value=1),

],p=1)
def load_paired_data(df, dir_prefix, augmentation=None):

    img_id = df.index.unique()[0]



    try:

        image = preprocess_input( skiImgRead( os.path.join(dir_prefix, img_id) ) )

    except:

        image = preprocess_input( np.zeros((IMG_HW, IMG_HW, 3), dtype=np.uint8) )



    mask = np.zeros((IMG_HW, IMG_HW, 1))

    for _,mask_rle in df['EncodedPixels'].iteritems():

        if mask_rle is np.nan:

            continue

        mask[:,:,0] += rle_decode(mask_rle)



    if augmentation:

        augmented = augmentation(image=image, mask=mask)

        image = augmented['image']

        mask  = augmented['mask']

    

    image = resize(image, (ZOOM_HW,ZOOM_HW))

    mask  = resize(mask.reshape(IMG_HW,IMG_HW), (ZOOM_HW,ZOOM_HW)).reshape((ZOOM_HW,ZOOM_HW,1))

    return image, mask
def batch_data_gen(csv_df, dir_prefix, batch_size, augmentation=None):

    name_idx_df = csv_df.set_index('ImageId')



#     img_ids = name_idx_df.index.unique().to_numpy()

    img_ids = np.array( name_idx_df.index.unique().tolist() )



    n_imgs  = img_ids.shape[0]

    

    while True:

        np.random.shuffle(img_ids)

        for idx in range(0, n_imgs, batch_size):

            batch_x = np.zeros( (batch_size,) + (ZOOM_HW, ZOOM_HW, 3) )

            batch_y = np.zeros( (batch_size,) + (ZOOM_HW, ZOOM_HW, 1) )



            end_idx = idx + batch_size

            batch_img_ids = img_ids[idx:end_idx]

            

            for i,img_id in enumerate(batch_img_ids):

                img_df = name_idx_df.loc[[img_id]]

                x, y = load_paired_data(img_df, dir_prefix, augmentation=augmentation)

                batch_x[i] += x

                batch_y[i] += y

            

            yield batch_x, batch_y
model = Unet(

    BACKBONE, 

    encoder_weights='imagenet',

    classes=1, 

    activation='sigmoid', 

    input_shape=(ZOOM_HW, ZOOM_HW, 3),

    decoder_filters=(128, 64, 32, 16, 8),

)

model.compile(optimizer='Adam', loss=bce_dice_loss, metrics=[iou_score])

BATCH_SIZE = 32
checkpoint = ModelCheckpoint(

    filepath='./best_model.h5', 

    monitor='val_iou_score', mode='max', 

    save_best_only=True, save_weights_only=True, 

    verbose=1

)



reduce_lr  = ReduceLROnPlateau(

    monitor='val_loss', mode='min', 

    factor=0.3, patience=3, min_lr=0.00001, 

    verbose=1

)
history = model.fit_generator(

    generator        = batch_data_gen(b_train_csv, TRAIN_DIR, BATCH_SIZE, augmentation=None), 

    validation_data  = batch_data_gen(b_valid_csv, TRAIN_DIR, BATCH_SIZE), 

    validation_steps = 50,

    steps_per_epoch  = 500,

    epochs           = 20,

    verbose = 2,

    callbacks=[ checkpoint, reduce_lr ]

)
sub_csv = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])



for test_id in tqdm( os.listdir( TEST_DIR ) ):

    fp  = os.path.join( TEST_DIR, test_id )

    img = skiImgRead(fp)

    assert ( img.shape == (IMG_HW, IMG_HW, 3) ), 'Bad Shape in image: "{}"'.format(fp)



    img = resize(img, (ZOOM_HW, ZOOM_HW))



    # TTA

    imgTTA1 = preprocess_input(img).reshape(1, ZOOM_HW, ZOOM_HW, 3)

    

    imgTTA1 = imgTTA1[:, :: 1, :: 1, :]

    imgTTA2 = imgTTA1[:, :: 1, ::-1, :]

    imgTTA3 = imgTTA1[:, ::-1, :: 1, :]

    imgTTA4 = imgTTA1[:, ::-1, ::-1, :]

    

    (rTTA1,rTTA2,rTTA3,rTTA4) = model.predict( np.concatenate( [imgTTA1, imgTTA2, imgTTA3, imgTTA4] ) )[:,:,:,0]



    result = (

        rTTA1[:: 1, :: 1] + 

        rTTA2[:: 1, ::-1] + 

        rTTA3[::-1, :: 1] + 

        rTTA4[::-1, ::-1]

    )/4

    

    result = resize(result, (IMG_HW, IMG_HW))

    labels = label( (result>0.5)+0 )



    

    # No Ship

    if labels.max() == 0:

        sub_csv = sub_csv.append({'ImageId':test_id}, ignore_index=True)

    else:

        for k in np.unique(labels[labels>0]):

            sub_csv = sub_csv.append(

                {

                    'ImageId'      : test_id, 

                    'EncodedPixels': rle_encode(labels==k)

                }, ignore_index=True)

sub_csv.to_csv('submission.csv', index=False)