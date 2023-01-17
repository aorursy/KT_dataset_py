import os
import numpy as np
from os import listdir
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.io import imsave, imread

data_path = '../input/'

image_rows = 420
image_cols = 580
def create_train_test_data():
    
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = int(len(images) // 2)

    print (total)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    print('Loading train data...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        #imgs[i, :, :, 0] = img
        #imgs_mask[i, :, :, 0] = img_mask
        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    
    imgs_train, imgs_test, imgs_mask_train, imgs_mask_test = train_test_split(imgs, imgs_mask, test_size=0.15, random_state=42)
    
    return imgs_train, imgs_mask_train,imgs_test, imgs_mask_test

if __name__ == '__main__':
    imgs_train, imgs_mask_train, imgs_test, imgs_mask_test = create_train_test_data() 
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

img_rows = 96
img_cols = 96
smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p
def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.3)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.3)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool5)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(conv6), conv5], axis=3)
    conv7 = Conv2D(1024, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Dropout(0.3)(conv7)
    conv7 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv7), conv4], axis=3)
    conv8 = Conv2D(512, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Dropout(0.3)(conv8)
    conv8 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv9 = Conv2D(256, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Dropout(0.3)(conv9)
    conv9 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv9)

    up10 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv10 = Conv2D(128, (3, 3), activation='relu', padding='same')(up10)
    conv10 = Dropout(0.3)(conv10)
    conv10 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv10)
    
    up11 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv11 = Conv2D(64, (3, 3), activation='relu', padding='same')(up11)
    conv11 = Dropout(0.3)(conv11)
    conv11 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv11)

    up12 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv13), conv2], axis=3)
    conv12 = Conv2D(32, (3, 3), activation='relu', padding='same')(up12)
    conv12 = Dropout(0.3)(conv12)
    conv12 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv12)
    
    conv13 = Conv2D(1, (1, 1), activation='sigmoid')(conv12)

    model = Model(inputs=[inputs], outputs=[conv12])

    model.compile(optimizer=Adam(lr=.00045), loss=dice_coef_loss, metrics=[dice_coef])

    return model

from keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(imgs_train):
    imgs_mask_train=imgs_train
    print('-'*30)
    print('Preprocessing train data...')
    print('-'*30)
    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train) 
    std = np.std(imgs_train)
    
    imgs_train -= mean
    imgs_train /= std
    
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights_best.h5', monitor='val_loss', save_best_only=True, verbose = 1)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    history = model.fit(imgs_train, imgs_mask_train, batch_size=256, epochs=50, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])
    print('-'*30)

    print(history.history.keys())
    
    import matplotlib.pyplot as plt
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model dice coefficient')
    plt.ylabel('dice coef')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('model_dice.png',bbox_inches='tight')
    
    return

train_model(imgs_train)