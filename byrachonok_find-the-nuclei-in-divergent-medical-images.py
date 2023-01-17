import random, os, sys
import numpy as np
import pandas as pd

import skimage.io
import skimage.transform
import skimage.filters
from skimage.feature import corner_peaks
from skimage.morphology import label

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from matplotlib import pyplot as plt
random.seed = 42
np.random.seed = 42
IMG_WIDTH = 192
IMG_HEIGHT = 192
N_CHANEL = 3
def get_images_and_masks_id(path='../input/stage1_train/'):
    train_id = next(os.walk(path))[1]
    list_train_set = []
    for i in train_id:
        list_train_set.append({
            'image_id':i,
            'masks_id':next(os.walk('{0}/{1}/masks/'.format(path, i)))[2]})
    return list_train_set
def stack_masks(masks_np_list):
    masks = np.mean(masks_np_list.transpose(1,2,0), axis=2)
    return masks.reshape(masks.shape[0], masks.shape[1], 1)
def get_image_and_masks(image_id, list_masks_id, path='../input/stage1_train/'):
    image_file = "{0}/{1}/images/{1}.png".format(path, image_id)
    image = skimage.io.imread(image_file)
    masks_list = []
    for mask_id in list_masks_id:
        mask_file = "{0}/{1}/masks/{2}".format(path, image_id, mask_id)
        masks_list.append(skimage.io.imread(mask_file))
    return {'image':image, 'masks':stack_masks(np.array(masks_list))}
def img_resize(img, output_shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANEL)):
    if img.shape[0] == output_shape[0] and img.shape[1] == output_shape[1]:
        return img[:,:,:N_CHANEL] / 255
    else:
        return skimage.transform.resize(
            img[:,:,:N_CHANEL], 
            output_shape,
            mode='reflect')
def mask_resize(mask, output_shape=(IMG_HEIGHT, IMG_WIDTH, 1)):
    if mask.shape[0] == output_shape[0] and mask.shape[1] == output_shape[1]:
        return (mask != 0).astype(np.uint)
    else:
        result = skimage.transform.resize(
            mask[:,:,:1], 
            output_shape,
            mode='reflect')
        result = (result > 0.5).astype(np.uint)
        return result
for image_and_masks_id in get_images_and_masks_id()[:10]:
    test_instance = get_image_and_masks(
        image_and_masks_id['image_id'], 
        image_and_masks_id['masks_id'])
    img = test_instance['image'] #img_resize
    mask = test_instance['masks'] #mask_resize
    plt.figure(1, figsize=(15,5))
    plt.subplot(1,2,1); plt.imshow(img)
    plt.subplot(1,2,2); plt.imshow(mask[:,:,0])
    plt.show()
input_data = []; output_data = []
for image_and_masks_id in get_images_and_masks_id():
    test_instance = get_image_and_masks(
        image_and_masks_id['image_id'], 
        image_and_masks_id['masks_id'])
    input_data.append(img_resize(test_instance['image']))
    output_data.append(mask_resize(test_instance['masks']))
input_data = np.array(input_data)
output_data = np.array(output_data)
assert input_data.shape[0] == output_data.shape[0]
from scipy import ndimage
from skimage.transform import swirl
import skimage.util
from random import randrange, uniform
def swirl_generator(image, mask):
    img = image[:,:,:3]
    msk = mask[:,:,0]
    
    for i in range(randrange(5,10)):
        r_radius = randrange(int(image.shape[0]/2), int(image.shape[0]/1.5))
        r_center_x = randrange(0, image.shape[0])
        r_center_y = randrange(0, image.shape[1])

        img = swirl(
            img.astype(np.float64), 
            center=(r_center_x, r_center_y), strength=1, radius=r_radius, mode='constant')
        msk = swirl(
            msk.astype(np.float64), 
            center=(r_center_x, r_center_y), strength=1, radius=r_radius, mode='constant')
    msk = (msk > 0.5).astype(np.uint)
    return img, msk
def gaus_filter(img):
    sigma = uniform(0.5, 2.5)
    return ndimage.gaussian_filter(img, sigma)
def rand_noise(img):
    sigma = uniform(0, 0.03)
    return skimage.util.random_noise(img, var=sigma)
# test swirl_generator
fig, ax = plt.subplots(2,4, figsize=(14,8))

plots = []; num = 100
plots.append((input_data[num], np.squeeze(output_data[num])))
plots.append((swirl_generator(input_data[num], output_data[num])))
plots.append((gaus_filter(input_data[num]), output_data[num][:,:,0]))
plots.append((rand_noise(input_data[num]), output_data[num][:,:,0]))

for i, p in enumerate(plots):
    ax[0][i].imshow(p[0])
    ax[1][i].imshow(p[1])
from skimage.transform import rotate
X = []; y = []
for img, msk in zip(input_data, output_data):
    for i in range(randrange(0,2)):
        r_angle = randrange(0, 360)
        rot_img = rotate(img.astype(np.float64), angle=r_angle)
        rot_msk = rotate(msk.astype(np.float64), angle=r_angle)
        if np.max(rot_msk) < 0.5: continue
        X.append(gaus_filter(np.fliplr(rot_img))) # vertical flip + gaus filte
        y.append(np.fliplr(rot_msk))
        X.append(rand_noise(np.flipud(rot_img))) # horisontal flip + noise
        y.append(np.flipud(rot_msk))
    X.append(img)
    y.append(msk)
X = np.array(X)
y = np.array(y)
del input_data, output_data
X.shape
y.shape
indexes = np.arange(X.shape[0])
np.random.shuffle(indexes)
train_indexes = indexes
# test normalization
assert 1.0 >= np.max(X) and np.min(X) <= 0.0
assert 1.0 >= np.max(y) and np.min(y) <= 0.0

y_max_list = np.max(y, axis=(1,2)) # find max in mask
assert (y_max_list < 0.5).any() == False
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.layers.core import Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        Y_negatives = tf.subtract(1.,y_pred)
        tp = tf.reduce_sum(tf.reduce_sum(tf.multiply(y_true,y_pred),1),1)  # True positives (i.e. the intersection)
        fp = tf.reduce_sum(tf.reduce_sum(tf.multiply(y_true,Y_negatives),1),1)  # False positives
        fn = tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.subtract(y_true,y_pred),y_pred),1),1)  # False negatives
        p = tp / (tp + fp + fn)
        prec.append(p)
    return K.mean(K.stack(prec), axis=0)
def create_model():
    print('model creat')
    k = 4
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, N_CHANEL))

    conv1 = Conv2D(2*k, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(2*k, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(4*k, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(4*k, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(8*k, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(8*k, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(16*k, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(16*k, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(32*k, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(32*k, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(16*k, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(16*k, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(8*k, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(8*k, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(4*k, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(4*k, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(2*k, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(2*k, (3, 3), activation='relu', padding='same')(conv9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
def compile_model(model):
    model.compile(
        optimizer='adam', 
        loss = 'binary_crossentropy',
        metrics=[mean_iou])
    return model
def fit_model(model, batch_size, validation_split, epochs):
    checkpointer = ModelCheckpoint('U-Net_model.h5', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(patience=5, verbose=1)
    history = model.fit(
        X[train_indexes], 
        y[train_indexes], 
        validation_split=validation_split, 
        batch_size=batch_size, 
        epochs=epochs,
        callbacks=[checkpointer, earlystopper])
    return history
#delete old model
!rm ./U-Net_model.h5
%%time
from keras.models import load_model
if os.path.isfile('../working/U-Net_model.h5'):
    print('model load')
    model = load_model('../working/U-Net_model.h5',  custom_objects={'mean_iou': mean_iou})
else:
    model = create_model()
model = compile_model(model)
fit_model(model, 4, 0.1, 12)
del X, y
# load test dataset
test_img_name = [img for img in next(os.walk('../input/stage1_test/'))[1]]
test_img = [skimage.io.imread('../input/stage1_test/{0}/images/{0}.png'.format(img)) for img in test_img_name]
input_shape = [img.shape[:2] for img in test_img]
test_img = [img_resize(img) for img in test_img] # вернуть размеры изображений назад
test_img = np.array(test_img)
test_img.shape
assert 1.0 >= np.max(test_img) and np.min(test_img) <= 0.0
model = load_model('../working/U-Net_model.h5',  custom_objects={'mean_iou': mean_iou})
preds_test= model.predict(test_img, verbose=1)
def return_shape_masks(masks, shapes):
    reshape_mask = []
    for i, shape in enumerate(shapes):
        mask = skimage.transform.resize(
            masks[i], 
            shape, 
            mode='reflect')
        mask = (mask > 0.5).astype(np.uint8)
        reshape_mask.append(mask)
    return np.array(reshape_mask)

def return_shape_img(images, shapes):
    reshape_images = []
    for i, shape in enumerate(shapes):
        img = skimage.transform.resize(
            images[i],
            (shape[0], shape[1], 3),
            mode='reflect')
        reshape_images.append(img)
    return np.array(reshape_images)
from skimage.morphology import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

def rle_encoding(x): # функция находит все точки на изображении
    '''x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list'''
    dots = np.where(x.T.flatten() > 0.5)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.50): # функция разделяет пятна на группы
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
images = return_shape_img(test_img, input_shape)
masks = return_shape_masks(preds_test, input_shape)

for i, mask in enumerate(masks[:10]):
    plt.figure(figsize = (14, 5))
    plt.subplot(1, 2, 1); plt.imshow(images[i]);
    plt.subplot(1, 2, 2); plt.imshow(mask[:,:,0])
    plt.show()
new_test_ids = []
rles = []
for i, id in enumerate(test_img_name):
    rle = list(prob_to_rles(masks[i][:,:,0]))
    rles.extend(rle)
    new_test_ids.extend([id] * len(rle))
submission_df = pd.DataFrame()
submission_df['ImageId'] = new_test_ids
submission_df['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
submission_df.to_csv('submission.csv', index=False)
def rls_decode(text_line, out_shape):
    img = np.zeros(out_shape[0]*out_shape[1])
    
    str_numbers = text_line.split()
    
    int_numbers = []
    for str_nb in str_numbers:
        int_numbers.append(int(str_nb))
    
    pixcels = []; lengs = []
    for i, num in enumerate(int_numbers):
        if (i % 2 == 0) or (i == 0):
            pixcels.append(num)
        if (i % 2 == 1) or (i == 1):
            lengs.append(num)
            
    for px, l in zip(pixcels, lengs):
        for x in range(l):
            img[px+x] = 1
    return img
test_mask_id = '0999dab07b11bc85fb8464fc36c947fbd8b5d6ec49817361cb780659ca805eac'
test_mask_shape = (253, 519)
n_nuclei = submission_df['EncodedPixels'][submission_df['ImageId'] == test_mask_id].shape[0]

decode_masks = []
for i in range(n_nuclei):
    nucl = submission_df['EncodedPixels'][submission_df['ImageId'] == test_mask_id].iloc[i]
    nucl_mask = rls_decode(nucl, test_mask_shape)
    decode_masks.append(nucl_mask.reshape(test_mask_shape))
    
decode_masks = [e*i for i, e in enumerate(decode_masks)]
image = np.sum(np.array(decode_masks), axis=0)
plt.imshow(image)
print('count nuclei:',np.max(image))
