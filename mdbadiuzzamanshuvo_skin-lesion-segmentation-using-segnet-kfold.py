import tensorflow as tf

from skimage.color import rgb2gray

from tensorflow.keras import Input

from tensorflow.keras.models import Model, load_model, save_model

from tensorflow.keras.layers import Input, Activation, BatchNormalization, UpSampling2D, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, Dense, concatenate, Reshape

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator



import numpy as np

import pandas as pd

import glob

import PIL

from PIL import Image

import matplotlib.pyplot as plt

import cv2

%matplotlib inline



from sklearn.model_selection import train_test_split, KFold

from warnings import filterwarnings



filterwarnings('ignore')

np.random.seed(101)
import re

numbers = re.compile(r'(\d+)')

def numericalSort(value):

    parts = numbers.split(value)

    parts[1::2] = map(int, parts[1::2])

    return parts
filelist_trainx = sorted(glob.glob('../input/*/trainx/*.bmp'), key=numericalSort)

X_train = np.array([np.array(Image.open(fname)) for fname in filelist_trainx])



filelist_trainy = sorted(glob.glob('../input/*/trainy/*.bmp'), key=numericalSort)

Y_train = np.array([np.array(Image.open(fname)) for fname in filelist_trainy])
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size = 0.25, random_state = 101)
plt.figure(figsize=(20,9))

plt.subplot(2,4,1)

plt.imshow(X_train[0])

plt.subplot(2,4,2)

plt.imshow(X_train[3])

plt.subplot(2,4,3)

plt.imshow(X_train[54])

plt.subplot(2,4,4)

plt.imshow(X_train[77])

plt.subplot(2,4,5)

plt.imshow(X_train[100])

plt.subplot(2,4,6)

plt.imshow(X_train[125])

plt.subplot(2,4,7)

plt.imshow(X_train[130])

plt.subplot(2,4,8)

plt.imshow(X_train[149])

plt.show()
plt.figure(figsize=(20,9))

plt.subplot(2,4,1)

plt.imshow(Y_train[0], cmap = plt.cm.binary_r)

plt.subplot(2,4,2)

plt.imshow(Y_train[3], cmap = plt.cm.binary_r)

plt.subplot(2,4,3)

plt.imshow(Y_train[54], cmap = plt.cm.binary_r)

plt.subplot(2,4,4)

plt.imshow(Y_train[77], cmap = plt.cm.binary_r)

plt.subplot(2,4,5)

plt.imshow(Y_train[100], cmap = plt.cm.binary_r)

plt.subplot(2,4,6)

plt.imshow(Y_train[125], cmap = plt.cm.binary_r)

plt.subplot(2,4,7)

plt.imshow(Y_train[130], cmap = plt.cm.binary_r)

plt.subplot(2,4,8)

plt.imshow(Y_train[149], cmap = plt.cm.binary_r)

plt.show()
def jaccard_distance(y_true, y_pred, smooth=100):

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)

    sum_ = K.sum(K.square(y_true), axis = -1) + K.sum(K.square(y_pred), axis=-1)

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return (1 - jac)
def iou(y_true, y_pred, smooth = 100):

    intersection = K.sum(K.abs(y_true * y_pred))

    sum_ = K.sum(K.square(y_true)) + K.sum(K.square(y_pred))

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return jac
def dice_coe(y_true, y_pred, smooth = 100):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):

    return -dice_coe(y_true, y_pred)
def precision(y_true, y_pred):

    '''Calculates the precision, a metric for multi-label classification of

    how many selected items are relevant.

    '''

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision
def recall(y_true, y_pred):

    '''Calculates the recall, a metric for multi-label classification of

    how many relevant items are selected.

    '''

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall
def accuracy(y_true, y_pred):

    '''Calculates the mean accuracy rate across all predictions for binary

    classification problems.

    '''

    return K.mean(K.equal(y_true, K.round(y_pred)))
def random_rotation(x_image, y_image):

    rows_x,cols_x, chl_x = x_image.shape

    rows_y,cols_y = y_image.shape

    rand_num = np.random.randint(-40,40)

    M1 = cv2.getRotationMatrix2D((cols_x/2,rows_x/2),rand_num,1)

    M2 = cv2.getRotationMatrix2D((cols_y/2,rows_y/2),rand_num,1)

    x_image = cv2.warpAffine(x_image,M1,(cols_x,rows_x))

    y_image = cv2.warpAffine(y_image.astype('float32'),M2,(cols_y,rows_y))

    return x_image, y_image.astype('int')



def horizontal_flip(x_image, y_image):

    x_image = cv2.flip(x_image, 1)

    y_image = cv2.flip(y_image.astype('float32'), 1)

    return x_image, y_image.astype('int')
def img_augmentation(x_train, y_train):

    x_rotat = []

    y_rotat = []

    x_flip = []

    y_flip = []

    for idx in range(len(x_train)):

        x,y = random_rotation(x_train[idx], y_train[idx])

        x_rotat.append(x)

        y_rotat.append(y)

        x,y = horizontal_flip(x_train[idx], y_train[idx])

        x_flip.append(x)

        y_flip.append(y)

    return np.array(x_rotat), np.array(y_rotat), np.array(x_flip), np.array(y_flip)
x_rotated, y_rotated, x_flipped, y_flipped = img_augmentation(x_train, y_train)
x_train_full = np.concatenate([x_train, x_rotated, x_flipped])

y_train_full = np.concatenate([y_train, y_rotated, y_flipped])
img_num = 7

plt.figure(figsize=(12,12))

plt.subplot(3,2,1)

plt.imshow(x_train_full[img_num])

plt.title('Original Image')

plt.subplot(3,2,2)

plt.imshow(y_train_full[img_num], plt.cm.binary_r)

plt.title('Original Mask')

plt.subplot(3,2,3)

plt.imshow(x_train_full[img_num+1])

plt.title('Rotated Image')

plt.subplot(3,2,4)

plt.imshow(y_train_full[img_num+1], plt.cm.binary_r)

plt.title('Rotated Mask')

plt.subplot(3,2,5)

plt.imshow(x_train_full[img_num+2])

plt.title('Flipped Image')

plt.subplot(3,2,6)

plt.imshow(y_train_full[img_num+2], plt.cm.binary_r)

plt.title('Flipped Mask')

plt.show()
#x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size = 0.20, random_state = 101)

kf = KFold(n_splits = 5, shuffle=False)
# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)

INPUT_CHANNELS = 3

# Number of output masks (1 in case you predict only one type of objects)

OUTPUT_MASK_CHANNELS = 1

# Pretrained weights

def segnet(input_size=(512, 512, 1)):



    # Encoding layer

    img_input = Input(input_size)

    x = Conv2D(64, (3, 3), padding='same', name='conv1',strides= (1,1))(img_input)

    x = BatchNormalization(name='bn1')(x)

    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='conv2')(x)

    x = BatchNormalization(name='bn2')(x)

    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    

    x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)

    x = BatchNormalization(name='bn3')(x)

    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='conv4')(x)

    x = BatchNormalization(name='bn4')(x)

    x = Activation('relu')(x)

    x = MaxPooling2D()(x)



    x = Conv2D(256, (3, 3), padding='same', name='conv5')(x)

    x = BatchNormalization(name='bn5')(x)

    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='conv6')(x)

    x = BatchNormalization(name='bn6')(x)

    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='conv7')(x)

    x = BatchNormalization(name='bn7')(x)

    x = Activation('relu')(x)

    x = MaxPooling2D()(x)



    x = Conv2D(512, (3, 3), padding='same', name='conv8')(x)

    x = BatchNormalization(name='bn8')(x)

    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='conv9')(x)

    x = BatchNormalization(name='bn9')(x)

    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='conv10')(x)

    x = BatchNormalization(name='bn10')(x)

    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    

    x = Conv2D(512, (3, 3), padding='same', name='conv11')(x)

    x = BatchNormalization(name='bn11')(x)

    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='conv12')(x)

    x = BatchNormalization(name='bn12')(x)

    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='conv13')(x)

    x = BatchNormalization(name='bn13')(x)

    x = Activation('relu')(x)

    x = MaxPooling2D()(x)



    x = Dense(256, activation = 'relu', name='fc1')(x)

    x = Dense(256, activation = 'relu', name='fc2')(x)

    # Decoding Layer 

    x = UpSampling2D()(x)

    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv1')(x)

    x = BatchNormalization(name='bn14')(x)

    x = Activation('relu')(x)

    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv2')(x)

    x = BatchNormalization(name='bn15')(x)

    x = Activation('relu')(x)

    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv3')(x)

    x = BatchNormalization(name='bn16')(x)

    x = Activation('relu')(x)

    

    x = UpSampling2D()(x)

    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv4')(x)

    x = BatchNormalization(name='bn17')(x)

    x = Activation('relu')(x)

    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv5')(x)

    x = BatchNormalization(name='bn18')(x)

    x = Activation('relu')(x)

    x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv6')(x)

    x = BatchNormalization(name='bn19')(x)

    x = Activation('relu')(x)



    x = UpSampling2D()(x)

    x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv7')(x)

    x = BatchNormalization(name='bn20')(x)

    x = Activation('relu')(x)

    x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv8')(x)

    x = BatchNormalization(name='bn21')(x)

    x = Activation('relu')(x)

    x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv9')(x)

    x = BatchNormalization(name='bn22')(x)

    x = Activation('relu')(x)



    x = UpSampling2D()(x)

    x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv10')(x)

    x = BatchNormalization(name='bn23')(x)

    x = Activation('relu')(x)

    x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv11')(x)

    x = BatchNormalization(name='bn24')(x)

    x = Activation('relu')(x)

    

    x = UpSampling2D()(x)

    x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv12')(x)

    x = BatchNormalization(name='bn25')(x)

    x = Activation('relu')(x)

    x = Conv2DTranspose(1, (3, 3), padding='same', name='deconv13')(x)

    x = BatchNormalization(name='bn26')(x)

    x = Activation('sigmoid')(x)

    pred = Reshape((input_size[0], input_size[1]))(x)

    

    return Model(inputs=img_input, outputs=pred)
losses = []

accuracies = []

ious = []

dice_cos = []

precisions = []

recalls = []

histories = []



for j, (train_idx, val_idx) in enumerate(kf.split(x_train_full, y_train_full)):

    

    print('\nFold ',j+1)

    X_train_cv = x_train_full[train_idx]

    y_train_cv = y_train_full[train_idx]

    X_valid_cv = x_train_full[val_idx]

    y_valid_cv= y_train_full[val_idx]

    

    model = segnet(input_size = (224, 224, INPUT_CHANNELS))



    model.compile(optimizer= Adam(lr=1e-3), loss= [dice_coef_loss]

              , metrics=[iou, dice_coe, precision, recall, accuracy])

    

    model_checkpoint = ModelCheckpoint(str(j+1) + '_skin_leison.hdf5', 

                                       verbose=1, 

                                       save_best_only=True)

    

    callbacks_list = [model_checkpoint]

    history = model.fit(X_train_cv,

                     y_train_cv,

                     epochs= 35,

                     callbacks = callbacks_list,

                     batch_size= 1,

                     validation_data=(X_valid_cv, y_valid_cv))

    

    model = load_model(str(j+1) + '_skin_leison.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss,

                                                                           'iou': iou, 'precision': precision, 'recall': recall,

                                                                           'accuracy': accuracy, 'dice_coe': dice_coe})



    results = model.evaluate(X_valid_cv, y_valid_cv)

    results = dict(zip(model.metrics_names,results))

                   

    accuracies.append(results['accuracy'])

    losses.append(results['loss'])

    ious.append(results['iou'])

    dice_cos.append(results['dice_coe'])

    precisions.append(results['precision'])

    recalls.append(results['recall'])

    histories.append(history)
print('average accuracy : ', np.mean(np.array(accuracies)), '+-', np.std(np.array(accuracies)))

print('average loss : ', np.mean(np.array(losses)), '+-', np.std(np.array(losses)))

print('average iou : ', np.mean(np.array(ious)), '+-', np.std(np.array(ious)))

print('average dice_coe : ', np.mean(np.array(dice_cos)), '+-', np.std(np.array(dice_cos)))

print('average precision : ', np.mean(np.array(precisions)), '+-', np.std(np.array(precisions)))

print('average recall : ', np.mean(np.array(recalls)), '+-', np.std(np.array(recalls)))
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
model = load_model('1_skin_leison.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss,

                                                           'iou': iou, 'precision': precision, 'recall': recall,

                                                           'accuracy': accuracy, 'dice_coe': dice_coe})
for i in range(10):

    index=np.random.randint(1,len(x_train_full))

    pred=model.predict(x_train_full[index][np.newaxis, :, :, :])



    plt.figure(figsize=(12,12))

    plt.subplot(1,3,1)

    plt.imshow(x_train_full[index])

    plt.title('Original Image')

    plt.subplot(1,3,2)

    plt.imshow(np.squeeze(y_train_full[index]))

    plt.title('Original Mask')

    plt.subplot(1,3,3)

    plt.imshow(np.squeeze(pred) > .5)

    plt.title('Predicted mask')

    plt.show()
