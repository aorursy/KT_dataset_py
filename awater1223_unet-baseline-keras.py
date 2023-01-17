!pip install git+https://github.com/qubvel/segmentation_models

!pip install albumentations



import os, sys, gc



import pandas as pd

import numpy  as np



import seaborn as sns

import matplotlib.pyplot as plt



from tqdm.auto       import tqdm

from multiprocessing import Pool, cpu_count



from cv2        import resize

from skimage.io import imread as skiImgRead



from sklearn.model_selection import KFold, train_test_split



from segmentation_models           import Unet, get_preprocessing

from segmentation_models.utils     import set_trainable

from segmentation_models.losses    import DiceLoss, BinaryCELoss, BinaryFocalLoss, JaccardLoss

from segmentation_models.metrics   import IOUScore, FScore



import keras.backend as K



from keras.utils     import Sequence

from keras.models    import load_model

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
IMG_H  = 256

IMG_W  = 1600



ZOOM_H = 128

ZOOM_W = 800



BACKBONE = 'seresnext50'

preprocess_input = get_preprocessing(BACKBONE)



DATA_DIR  = '../input'

TRAIN_DIR = 'train_images'

TEST_DIR  = 'test_images'
def rle_decode(rle_mask):

    '''

    rle_mask: run-length as string formated (start length)

    shape: (height,width) of array to return

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = rle_mask.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(IMG_W*IMG_H, dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(IMG_W,IMG_H).T



def rle_encode(im):

    '''

    im: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels = im.flatten(order = 'F')

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



# from https://www.kaggle.com/robertkag/rle-to-mask-converter

def rle_to_mask(rle_string,height,width):

    '''

    convert RLE(run length encoding) string to numpy array



    Parameters: 

    rleString (str): Description of arg1 

    height (int): height of the mask

    width (int): width of the mask 



    Returns: 

    numpy.array: numpy array of the mask

    '''

    rows, cols = height, width

    if rle_string == -1:

        return np.zeros((height, width))

    else:

        rleNumbers = [int(numstring) for numstring in rle_string.split(' ')]

        rlePairs = np.array(rleNumbers).reshape(-1,2)

        img = np.zeros(rows*cols,dtype=np.uint8)

        for index,length in rlePairs:

            index -= 1

            img[index:index+length] = 255

        img = img.reshape(cols,rows)

        img = img.T

        return img

    

# Thanks to the authors of: https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode

def mask_to_rle(mask):

    '''

    Convert a mask into RLE

    

    Parameters: 

    mask (numpy.array): binary mask of numpy array where 1 - mask, 0 - background



    Returns: 

    sring: run length encoding 

    '''

    pixels= mask.T.flatten()

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

    if v == ' -1' or v is np.nan or type(v) != str:

        return np.nan

    else:

        return rle_decode(v).sum()
train_csv  = pd.read_csv( os.path.join( DATA_DIR, 'train.csv') )



train_csv[['ImageId','Class']] = train_csv['ImageId_ClassId'].str.split('_',expand=True)



train_csv['Class']    = train_csv['Class'].astype(np.int)

train_csv['npixel']   = count_pix_inpool( train_csv['EncodedPixels'] )

train_csv['withMask'] = ~train_csv['npixel'].isnull()
sns.kdeplot(train_csv['npixel'])

plt.xscale('log')

plt.legend().set_visible(False)

plt.xlabel('# of pixel is mask')

plt.ylabel('Density')

plt.show()
sns.barplot( data=train_csv.dropna().groupby('Class').count().reset_index(), x='Class',y='ImageId' )

plt.ylabel('# of Images')

plt.show()
fig, ax = plt.subplots()



sns.boxplot(

    data=train_csv.dropna(),

    x = 'Class',

    y = 'npixel',

    ax = ax,

)

ax.set_yscale('log')



plt.show()
sns.barplot( 

    data = train_csv.dropna().groupby('ImageId').count().groupby('Class').count().reset_index(),

    x = 'Class',

    y = 'ImageId_ClassId'

)

plt.ylabel('# of Class in One Image')

plt.show()
plt.pie(

    x       = (train_csv.groupby('ImageId')['withMask'].any().value_counts()/train_csv['ImageId'].unique().shape[0]).values,

    labels  = (train_csv.groupby('ImageId')['withMask'].any().value_counts()/train_csv['ImageId'].unique().shape[0]).index,

    autopct = '%3.1f %%',

    shadow  = True,

    labeldistance = 1.1,

    startangle  = 90,

    pctdistance = 0.6

);

plt.title('Image with Mask or Not');

plt.show()
DROP_NO_MASK_FRACTION = 0.0



balanced_train_csv = (

    train_csv.set_index('ImageId')

    .drop(

        train_csv.set_index('ImageId').drop(

            train_csv['ImageId'].unique()[train_csv.groupby('ImageId')['withMask'].any()]

        ).sample( frac = DROP_NO_MASK_FRACTION ).index

    )

)
class ImgMaskGenerator(Sequence):



    def __init__(self, 

                 img_ids, data_dir, data_df,

                 image_shape  = (256,256,3), 

                 masks_channels = 1,

                 augmentor = None,

                 bg_tag = False,

                 batch_size=8, shuffle=True):

        '''

        Initialization

        '''

        if bg_tag:

            bg = 1

        else:

            bg = 0

        

        self.img_ids     = img_ids

        self.data_dir    = data_dir

        self.data_df     = data_df

        self.image_shape = image_shape

        self.masks_shape = image_shape[:2] + (masks_channels+bg,)

        self.batch_size  = batch_size

        self.augmentor   = augmentor,

        self.bg          = bg

        self.shuffle     = shuffle

        self.idxs        = np.arange(len(self.img_ids))

        self.on_epoch_end()



    def __len__(self):

        '''

        Denotes the number of batches per epoch

        '''

        return int(np.ceil(len(self.img_ids) / float(self.batch_size)))



    def __getitem__(self, idx, augmentation=False):

        '''

        Generate one batch of data

        '''

        

        end_idx = idx + self.batch_size

        

        batch_x = np.zeros( (self.batch_size,) + self.image_shape )

        batch_y = np.zeros( (self.batch_size,) + self.masks_shape )



        batch_img_ids = self.img_ids[idx:end_idx]

        

        for i,img_id in enumerate(batch_img_ids):

            if augmentation:

                x, y = self._load_paired_data(img_id, augmentation=self.augmentor)

            else:

                x, y = self._load_paired_data(img_id, augmentation=None)



            batch_x[i] += x

            batch_y[i] += y



        return batch_x, batch_y

    

    def on_epoch_end(self):

        '''

        Updates indexes after each epoch

        '''

        if self.shuffle == True:

            np.random.shuffle(self.idxs)

            

    def _load_paired_data(self, img_id, augmentation=None):

        height, width = self.image_shape[:2]

        

        img_fp = os.path.join( self.data_dir, img_id )

        image = skiImgRead(img_fp)

        image = resize(image, (width,height))

        masks = np.zeros( self.masks_shape )

        

        if self.bg == 1:

            masks = masks[:,:,:-1]

        



        for cls, row in self.data_df.loc[img_id].set_index('Class').iterrows():

            if row['EncodedPixels'] is np.nan:

                mask = np.zeros((height, width))

            else:

                mask = rle_decode(row['EncodedPixels'])

                mask = resize(mask, (width,height))



            masks[:,:,cls-1] += mask



        if self.bg == 1:

            bg_mask = np.ones(self.image_shape[:2]) - masks.max(axis=-1)

            masks = np.dstack([masks,bg_mask])

        

        if augmentation:

            augmented = augmentation(image=image, mask=masks)

            image = augmented['image']

            masks = augmented['mask']



        image = preprocess_input(image)

        return image, masks
from albumentations import (

    Compose, OneOf, ToFloat, PadIfNeeded, Resize, NoOp, 

    Flip, HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, Transpose,

    Blur, MotionBlur, MedianBlur, JpegCompression, Cutout,

    RandomCrop, RandomScale, RandomSizedCrop, CenterCrop,

    RandomContrast, RandomBrightness, RandomBrightnessContrast, CLAHE, RandomGamma,

    RGBShift, GaussNoise, HueSaturationValue,

    GridDistortion, ElasticTransform, OpticalDistortion,

    IAASharpen, IAAPiecewiseAffine, IAAAdditiveGaussianNoise

)



aug = Compose([

    OneOf([

        NoOp(),

        Flip(),

        HorizontalFlip(),

        VerticalFlip(),

    ], p=1), 

    

    OneOf([

        NoOp(),

        Blur(blur_limit=3),

        MedianBlur(blur_limit=3),

        MotionBlur(blur_limit=3),

        JpegCompression(),

    ], p=0.3),

    

    OneOf([

        NoOp(),

        RandomGamma(),

        RandomContrast(),

        RandomBrightness(),

        RandomBrightnessContrast(),

        CLAHE(),

     ], p=0.3),



    OneOf([

        NoOp(),

        GaussNoise(),

        HueSaturationValue(),

        Cutout(num_holes=8, max_h_size=4, max_w_size=8),

        IAAAdditiveGaussianNoise(),

    ], p=0.3),

    

    OneOf([

        NoOp(),

        OpticalDistortion(),

        GridDistortion(),

        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),

    ], p=0.2),

    

    OneOf([

        NoOp(),

        CenterCrop(height=ZOOM_H, width=ZOOM_W//2),

        RandomCrop(height=ZOOM_H, width=ZOOM_W//2),

        RandomScale(),

        RandomSizedCrop(min_max_height=(ZOOM_H/2, ZOOM_H), height=ZOOM_H, width=ZOOM_W),

    ],p=0.3),

    

    IAASharpen(p=0.2), 

    ShiftScaleRotate(rotate_limit=10, p=0.3),



    PadIfNeeded(min_height=ZOOM_H,min_width=ZOOM_W, p=1),

    Resize(height=ZOOM_H, width=ZOOM_W, p=1),

    ToFloat(max_value=1),

], p=1)
train_img_ids = np.array( balanced_train_csv.index.unique().tolist() )

train_img_ids.shape
total_gen = ImgMaskGenerator( 

    img_ids  = train_img_ids, 

    data_dir = os.path.join( DATA_DIR, TRAIN_DIR),

    data_df  = balanced_train_csv,

    image_shape  = (ZOOM_H,ZOOM_W,3), 

    masks_channels  = 4,

#     bg_tag=1,

    batch_size=8, shuffle=True

)



fig, axs = plt.subplots(ncols=2, nrows=5, figsize=(10, 5), sharex=True, sharey=True)



for i, img_id in enumerate( np.random.choice(train_img_ids, 5) ):

    x, y = total_gen._load_paired_data(img_id)

    

    axs[i,0].set_xlabel(img_id)

    axs[i,0].imshow(x)

    axs[i,1].imshow(np.sum(y,axis=-1))



axs[0,0].set_title('Input')

axs[0,1].set_title('Mask')



plt.xticks([])

plt.yticks([])



plt.show()



del total_gen;

gc.collect();
def dice_test(y_true, y_pred, smooth=1, axis=None):

    """Generate the 'Dice' coefficient for the provided prediction.

    Args:

        y_true: The expected/desired output mask.

        y_pred: The actual/predicted mask.

    Returns:

        The Dice coefficient between the expected and actual outputs. Values

        closer to 1 are considered 'better'.

    """

    if axis is None:

        y_true_f = y_true.flatten()

        y_pred_f = y_pred.flatten()

        intersection = np.sum(y_true_f * y_pred_f)

        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    else:

        intersection = np.sum(y_true * y_pred, axis=axis)

        dice = (2. * intersection + smooth) / (np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) + smooth)

        return np.mean(dice)



def dice_coef(y_true, y_pred, smooth=1, axis=None):

    """Generate the 'Dice' coefficient for the provided prediction.

    Args:

        y_true: The expected/desired output mask.

        y_pred: The actual/predicted mask.

    Returns:

        The Dice coefficient between the expected and actual outputs. Values

        closer to 1 are considered 'better'.

    """

    if axis is None:

        y_true_f = K.flatten(y_true)

        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)

        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    else:

        intersection = K.sum(y_true * y_pred, axis=axis)

        dice = (2. * intersection + smooth) / (K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + smooth)

        return K.mean(dice)
train_ids, holdout_ids = train_test_split(train_img_ids, random_state=42, test_size=0.1)
BATCH_SIZE = 16



train_gen = ImgMaskGenerator( 

    img_ids  = train_ids,

    data_dir = os.path.join( DATA_DIR, TRAIN_DIR ),

    data_df  = balanced_train_csv,

    image_shape    = (ZOOM_H,ZOOM_W,3), 

    masks_channels = 4,

    bg_tag         = False,

    augmentor      = None,

    batch_size=BATCH_SIZE, shuffle=True,

)



valid_gen = ImgMaskGenerator( 

    img_ids  = holdout_ids,

    data_dir = os.path.join( DATA_DIR, TRAIN_DIR ),

    data_df  = balanced_train_csv,

    image_shape    = (ZOOM_H,ZOOM_W,3), 

    masks_channels = 4,

    bg_tag         = False,

    augmentor      = None,

    batch_size=BATCH_SIZE, shuffle=True,

)

model = Unet(

    BACKBONE,

    encoder_weights='imagenet',

    classes=4,

    activation='sigmoid',

    input_shape=(ZOOM_H, ZOOM_W, 3),

)



model.compile(

    optimizer = 'Adam', 

    loss = BinaryCELoss() + DiceLoss(), # v2

    metrics = [

        IOUScore(      threshold=0.5, per_image=True),

        FScore(beta=2, threshold=0.5, per_image=True),

    ]

)



best_model_fp = './best_model.h5'



checkpoint = ModelCheckpoint(

    filepath=best_model_fp,

    monitor='val_f2-score', mode='max', 

    save_best_only=True, save_weights_only=False, 

    verbose=1

)

reduce_lr  = ReduceLROnPlateau(

    monitor='val_loss', mode='min', 

    factor=0.3, patience=8, min_lr=0.00001, 

    verbose=1

)

earlyStop  = EarlyStopping(

    monitor='val_f2-score', mode='max', 

    min_delta=0, patience=15, 

    verbose=1

)
gc.collect();
history = model.fit_generator(

    generator        = train_gen,

    validation_data  = valid_gen,

    steps_per_epoch  = len(train_gen),

    validation_steps = len(valid_gen)//2,

    epochs           = 40,

    callbacks        = [ checkpoint, reduce_lr, earlyStop ],



    use_multiprocessing = True,

    workers = 2,

    verbose = 2,

)
fig, axs = plt.subplots( nrows=1, ncols=3, figsize=(16,3) )



los       = model.history.history['loss']

vlos      = model.history.history['val_loss']

dicecoef  = model.history.history['f2-score']

vdicecoef = model.history.history['val_f2-score']

iou       = model.history.history['iou_score']

viou      = model.history.history['val_iou_score']



epochs = np.arange(1, len(los)+1)



axs[0].plot(epochs, los,       label='Training loss')

axs[0].plot(epochs, vlos,      label='Validation loss')

axs[1].plot(epochs, dicecoef,  label='dice_coef')

axs[1].plot(epochs, vdicecoef, label='Validation dice_coef')

axs[2].plot(epochs, iou,       label='IOU')

axs[2].plot(epochs, viou,      label='Validation IOU')



axs[0].legend()

axs[1].legend()

axs[2].legend()



plt.show()
model.load_weights('best_model.h5')
n = 10

fig, axs = plt.subplots(ncols=3, nrows=n, figsize=(10, n), sharex=True, sharey=True)



for i, img_id in enumerate( np.random.choice(holdout_ids, n) ):

    x, y = valid_gen._load_paired_data(img_id)

    yp = model.predict( np.expand_dims(x, axis=0) )

    ys = np.sum( (yp[0]>0.5)+0, axis=-1 )



    axs[i,0].set_ylabel(img_id, rotation=0, ha='right')

    axs[i,0].imshow(x)

    axs[i,1].imshow(np.sum(y,axis=-1))

    axs[i,2].imshow(ys)





axs[0,0].set_title('Input')

axs[0,1].set_title('Mask')

axs[0,2].set_title('Predict')



plt.xticks([])

plt.yticks([])



plt.subplots_adjust(wspace=0.05,hspace=0.001,bottom=0.05,top=0.5)



plt.show()
gc.collect();
dice_ts = {}



thresholds = np.arange(0.3, 1.0, 0.05)



for img_id in tqdm(holdout_ids):

    x, y = valid_gen._load_paired_data(img_id)

    yp = model.predict(np.expand_dims(x,axis=0))[0]

    

    for t in thresholds:

        yp = ((yp>t)+0)

        dice_t = dice_test(y[:,:,:4],yp[:,:,:4], axis=(0,1))



        if t not in dice_ts.keys():

            dice_ts[t] = []



        dice_ts[t].append(dice_t)

            

gc.collect();
for t in thresholds:

    dice_ts[t] = np.array( dice_ts[t] ).mean()



dice_ts = pd.Series(dice_ts)[thresholds].values
threshold_best_index = np.argmax(dice_ts) 

dice_best = dice_ts[threshold_best_index]

threshold_best = thresholds[threshold_best_index]



plt.plot(thresholds, dice_ts)

plt.plot(threshold_best, dice_best, "xr", label="Best threshold")

plt.xlabel("Threshold")

plt.ylabel("Dice")

plt.title("Threshold vs Dice ({}, {})".format(threshold_best, dice_best))

plt.legend()
n = 10

fig, axs = plt.subplots(ncols=3, nrows=n, figsize=(10, n), sharex=True, sharey=True)



for i, img_id in enumerate( np.random.choice(holdout_ids, n) ):

    x, y = valid_gen._load_paired_data(img_id)

    yp = model.predict( np.expand_dims(x, axis=0) )

    ys = np.sum( (yp[0]>threshold_best)+0, axis=-1 )



    axs[i,0].set_ylabel(img_id, rotation=0, ha='right')

    axs[i,0].imshow(x)

    axs[i,1].imshow(np.sum(y,axis=-1))

    axs[i,2].imshow(ys)





axs[0,0].set_title('Input')

axs[0,1].set_title('Mask')

axs[0,2].set_title('Predict')



plt.xticks([])

plt.yticks([])



plt.subplots_adjust(wspace=0.05,hspace=0.001,bottom=0.05,top=0.5)



plt.show()