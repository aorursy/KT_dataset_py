# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import os
from keras import layers
from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import matplotlib.pyplot as plt
from skimage.util import montage2d
from skimage.morphology import binary_opening, disk
from skimage.io import imread, imshow
from skimage.segmentation import mark_boundaries
import keras.backend as K
#create grid of images from rgb images
montage_rgb = lambda x: np.stack([montage2d(x[:,:,:,i]) for i in range(x.shape[3])],-1)
SAMPLE_PER_SHIP_COUNT=2000
TEST_PROP=0.2
BATCH_SIZE=16
MAX_TRAIN_EPOCHS=99
# downsampling in preprocessing
IMG_SCALING = (3, 3)

segmentations = pd.read_csv('../input/train_ship_segmentations.csv')
segmentations.head()
print ('total images in train data: {}'.format(segmentations['ImageId'].drop_duplicates().count()))
segmentations['has_ship']=segmentations['EncodedPixels'].apply(lambda x: 0 if pd.isnull(x) else 1)
segmentations['path']=segmentations['ImageId'].apply(lambda x: '../input/train/'+x)
ships_per_image=segmentations.groupby('ImageId').agg({'has_ship': 'sum'}).reset_index().rename(columns={'has_ship': 'ship_count'})
ships_per_image=ships_per_image.groupby('ship_count').apply(lambda x: x.sample(SAMPLE_PER_SHIP_COUNT) if len(x)> SAMPLE_PER_SHIP_COUNT else x)
ships_per_image.head()
sampled_set=pd.merge(segmentations, ships_per_image)
sample_segmentations = sampled_set[['ImageId','ship_count']].drop_duplicates()
sampled_set.head()
sampled_set_train_img, sampled_set_valid_img = train_test_split(sample_segmentations,  test_size = TEST_PROP, stratify = sample_segmentations['ship_count'],random_state=42)
sampled_set_train, sampled_set_valid = pd.merge(sampled_set, sampled_set_train_img), pd.merge(sampled_set, sampled_set_valid_img)
print ('training data shape: {}'.format(sampled_set_train.shape))
print ('test data shape: {}'.format(sampled_set_valid.shape))
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction #array of 

def multi_rle_decode(mask_rle_list):
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in mask_rle_list:
        if isinstance(mask,str):
            all_masks |= rle_decode(mask)
    return all_masks
def make_image_gen(df, batch_size=BATCH_SIZE):
    all_list = list(df.groupby('ImageId'))
    c_img = []
    c_mask= []
    while True:
        np.random.shuffle(all_list)
        for img_id, img_df in all_list:
            img_rgb_array = imread(img_df['path'].values[0])
            img_mask_array = multi_rle_decode(img_df['EncodedPixels'].values).reshape((768, 768, 1))
            if IMG_SCALING is not None:
                    img_rgb_array = img_rgb_array[::IMG_SCALING[0], ::IMG_SCALING[1]]
                    img_mask_array = img_mask_array[::IMG_SCALING[0], ::IMG_SCALING[1]]
            c_img += [img_rgb_array]
            c_mask += [img_mask_array]
            if len(c_img)>= batch_size:
                yield np.stack(c_img,0) , np.stack(c_mask,0)
                c_img = []
                c_mask= []
            
train_set_batches = make_image_gen(sampled_set_train)
train_x, train_y = next(train_set_batches)
print ('train x shape:{}'.format(train_x.shape))
print ('train y shape:{}'.format(train_y.shape))
train_y.max(), train_y.min(), train_x.max(), train_x.min()
np.unique(train_y) #all elements in response image are 0/1
valid_set_batches = make_image_gen(sampled_set_valid, 900)
valid_x, valid_y = next(valid_set_batches)
valid_x, valid_y = valid_x.astype('f')/255.0, valid_y.astype('f')/1.0
print ('valid x shape:{}'.format(valid_x.shape))
print ('valid y shape:{}'.format(valid_y.shape))
print('valid_x', valid_x.dtype, valid_x.min(), valid_x.max())
print('valid_y', valid_y.dtype, valid_y.min(), valid_y.max())
data_gen_args = dict(rotation_range = 45,
                      width_shift_range=0.1,
                      height_shift_range=0.1,
                      shear_range=0.02,
                      zoom_range=0.2,
                      horizontal_flip=True,
                      vertical_flip=True)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
def create_aug_data(train_set_batches, seed=1):
    for img, mask in train_set_batches:
        t_x = image_datagen.flow(img, batch_size=img.shape[0], shuffle=True, seed=seed)
        t_y = mask_datagen.flow(mask, batch_size=img.shape[0], shuffle=True, seed=seed)
        yield next(t_x)/255.0, next(t_y)
cur_gen = create_aug_data(train_set_batches)
t_x, t_y = next(cur_gen)
print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(40,20))
batch_rgb = montage_rgb(t_x)
batch_seg = montage2d(t_y[:, :, :, 0])
ax1.imshow(batch_rgb)
ax1.set_title('Images')
ax2.imshow(batch_seg)
ax2.set_title('Segmentations')
ax3.imshow(mark_boundaries(batch_rgb, batch_seg.astype(int),outline_color=(1,1,1)))
ax3.set_title('Outlined Ships')

def autoencoder(input_shape):
    #encoder
    input_img = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) 
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) 
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) 
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) 
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) 
        
    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) 
    up1 = UpSampling2D((2,2))(conv4) 
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) 
    up2 = UpSampling2D((2,2))(conv5) 
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) 
    
    model =  Model(inputs = input_img, outputs = decoded, name = 'Ship_model')
    
    return model
model = autoencoder(input_shape = t_x.shape[1:])
def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)
model.compile(optimizer='adam', loss=IoU, metrics=['binary_accuracy'])
model.summary()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="model_weights.best.hdf5"

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True, period=1)

early = EarlyStopping(monitor="val_loss", mode="min", verbose=2,
                      patience=10) # patience is number of epochs with no improvement after which training will be stopped

callbacks_list = [checkpoint, early]
aug_gen = create_aug_data(make_image_gen(sampled_set_train))
loss_history = [model.fit_generator(aug_gen,
                                 steps_per_epoch=7,#Total number of steps (batches of samples) to yield from generator before declaring one epoch finished 
                                 epochs=MAX_TRAIN_EPOCHS,
                                 validation_data=(valid_x, valid_y),
                                 callbacks=callbacks_list,
                                workers=1 # the generator is not very thread safe
                                           )]
IMG_SCALING
model.load_weights(weight_path)
model.save('model.h5')

epochs = np.concatenate([mh.epoch for mh in loss_history])
loss = np.concatenate([mh.history['loss'] for mh in loss_history])
val_loss  = np.concatenate([mh.history['val_loss'] for mh in loss_history])
train_accuracy = np.concatenate([mh.history['binary_accuracy'] for mh in loss_history])
test_accuracy = np.concatenate([mh.history['val_binary_accuracy'] for mh in loss_history])
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (30,10))

ax1.plot(epochs,train_accuracy, epochs,test_accuracy)
ax1.legend(['Training', 'Validation'])
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.set_title('accuracy train vs validation')

ax2.plot(epochs,loss, epochs,val_loss)
ax2.legend(['Training', 'Validation'])
ax2.set_xlabel('epoch')
ax2.set_ylabel('loss')
ax2.set_title('loss train vs validation')

if IMG_SCALING is not None:
    X_input = Input(shape=(None, None, 3))
    X = AveragePooling2D(IMG_SCALING)(X_input)
    X = model(X)
    X = UpSampling2D(IMG_SCALING)(X)
    fullres_model = Model(inputs = X_input, outputs = X)
else:
    fullres_model = model
fullres_model.save('fullres_model.h5')

fullres_model.summary()
#selecting batch size for display 
n=16
valid_img_set = pd.DataFrame(sampled_set_valid['ImageId'].sample(n))
valid_set = pd.merge(sampled_set,valid_img_set)
IMG_SCALING = None
valid_set_batches = make_image_gen(valid_set, n)
v_x, v_y = next(valid_set_batches)
print ('valid x shape:{}'.format(v_x.shape))
print ('valid y shape:{}'.format(v_y.shape))

v_img = []
v_mask= []
for i in range(n):
    rgb_img = v_x[i]/255.
    mask_img = v_y[i]/1.
    v_img += [rgb_img]
    v_mask += [mask_img]
v_img, v_mask = np.stack(v_img,0) , np.stack(v_mask,0)
print ('valid x shape:{}'.format(v_img.shape))
print ('valid y shape:{}'.format(v_mask.shape))

v_mask_pred = fullres_model.predict(v_img)
print ('valid y pred shape:{}'.format(v_mask_pred.shape))
def smooth(cur_seg):
    return binary_opening(cur_seg>0.999, np.expand_dims(disk(2), -1))
smooth_mask = []
for i in range(n):
    smooth_mask += [smooth(v_mask_pred[i])]
smooth_mask = np.stack(smooth_mask,0)
print ('smooth_mask shape:{}'.format(smooth_mask.shape))
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(40,20))
batch_rgb = montage_rgb(v_img)
batch_seg = montage2d(v_mask[:, :, :, 0])
batch_seg_pred = montage2d(smooth_mask[:, :, :, 0])
ax1.imshow(batch_rgb)
ax1.set_title('Images')
ax2.imshow(batch_seg)
ax2.set_title('actual')
ax3.imshow(batch_seg_pred)
ax3.set_title('pred')

t_y_pred = model.predict(t_x)
smooth_mask2 = []
for i in range(t_y_pred.shape[0]):
    smooth_mask2 += [smooth(t_y_pred[i])]
smooth_mask2 = np.stack(smooth_mask2,0)
print ('smooth_mask2 shape:{}'.format(smooth_mask2.shape))
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(40,20))
batch_rgb = montage_rgb(t_x)
batch_seg = montage2d(t_y[:, :, :, 0])
batch_seg_pred = montage2d(smooth_mask2[:, :, :, 0])
ax1.imshow(batch_rgb)
ax1.set_title('Images')
ax2.imshow(batch_seg)
ax2.set_title('actual')
ax3.imshow(batch_seg_pred)
ax3.set_title('pred')

segmentations_test = pd.read_csv('../input/test_ship_segmentations.csv')
segmentations_test.head()
#distinct images in test data
print ('Images in submission file: {}'.format(segmentations_test['ImageId'].drop_duplicates().count()))
test_df = segmentations_test.groupby('ImageId').apply(lambda x: list(x['EncodedPixels'].values)).reset_index().rename(columns={0: 'encode_list'})
test_df['EncodedPixels'] = test_df['encode_list'].apply(lambda x: rle_encode(multi_rle_decode(x)) if rle_encode(multi_rle_decode(x)) != '' else np.nan)
test_df['path'] = test_df['ImageId'].apply(lambda x: '../input/test/'+x)
test_df = test_df[['ImageId','EncodedPixels','path']]
test_df.head()
#test_df_sample = test_df.sample(frac=0.2).reset_index(drop=True)
def test_pred_gen(path):
    img_rgb_array = imread(path)
    img_rgb_array = img_rgb_array/255.
    img_rgb_array = np.expand_dims(img_rgb_array,0)
    c_pred = fullres_model.predict(img_rgb_array)
    c_ep_pred = rle_encode(smooth(c_pred[0])) if rle_encode(smooth(c_pred[0]))!='' else np.nan
    return c_ep_pred

#function check
test_pred_gen('../input/test/000532683.jpg')
#test_df_sample['encoded_pixel_pred'] = test_df_sample['path'].apply(lambda x: test_pred_gen(x))
test_df['encoded_pixel_pred'] = test_df['path'].apply(lambda x: test_pred_gen(x))
#test_df_sample.head()
test_df.head()
submit = test_df[['ImageId','encoded_pixel_pred']].rename(columns={'encoded_pixel_pred': 'EncodedPixels'})
submit.head()
submit.to_csv('submission_file.csv', index=False)
