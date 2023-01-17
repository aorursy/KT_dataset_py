from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
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

from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, add

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

        image_color_mode="rgb",

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
import pandas

from sklearn.model_selection import KFold



df = pandas.DataFrame(data={"filename": train_files, 'mask' : mask_files, 'dilate' : dilate_files})



kf = KFold(n_splits = 5, shuffle=False)
train_generator_args = dict(rotation_range=0.2,

                            width_shift_range=0.05,

                            height_shift_range=0.05,

                            shear_range=0.05,

                            zoom_range=0.05,

                            horizontal_flip=True,

                            fill_mode='nearest')



histories = []

losses = []

accuracies = []

dicecoefs = []

ious = []



BATCH_SIZE = 4

EPOCHS = 40



for k, (train_index, test_index) in enumerate(kf.split(df)):

    train_data_frame = df.iloc[train_index]

    test_data_frame = df.iloc[test_index]

    

    train_gen = train_generator(train_data_frame, BATCH_SIZE,

                                None,

                                train_generator_args,

                                target_size=(512,512))



    test_gener = train_generator(test_data_frame, BATCH_SIZE,

                                None,

                                train_generator_args,

                                target_size=(512,512))



    model = MultiResUnet(input_size=(512,512,3))

    model.compile(optimizer=Adam(lr=5e-6), loss=dice_coef_loss, \

                      metrics=[iou, dice_coef, 'binary_accuracy'])

    model.summary()



    model_checkpoint = ModelCheckpoint(str(k+1) + '_unet_lung_seg.hdf5', 

                                       monitor='loss', 

                                       verbose=1, 

                                       save_best_only=True)



    history = model.fit_generator(train_gen,

                                  steps_per_epoch=len(train_data_frame) / BATCH_SIZE, 

                                  epochs=EPOCHS, 

                                  callbacks=[model_checkpoint],

                                  validation_data = test_gener,

                                  validation_steps=len(test_data_frame) / BATCH_SIZE)

    

    model = load_model(str(k+1) + '_unet_lung_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})

    

    #test_gen = test_generator(test_files, target_size=(512,512))

    test_gen = train_generator(test_data_frame, BATCH_SIZE,

                                None,

                                train_generator_args,

                                target_size=(512,512))

    results = model.evaluate_generator(test_gen, steps=len(test_data_frame))

    results = dict(zip(model.metrics_names,results))

    

    histories.append(history)

    accuracies.append(results['binary_accuracy'])

    losses.append(results['loss'])

    dicecoefs.append(results['dice_coef'])

    ious.append(results['iou'])
print('accuracies : ', accuracies)

print('losses : ', losses)

print('dicecoefs : ', dicecoefs)

print('ious : ', ious)



print('-----------------------------------------------------------------------------')

print('-----------------------------------------------------------------------------')



print('average accuracy : ', np.mean(np.array(accuracies)))

print('average loss : ', np.mean(np.array(losses)))

print('average dicecoefs : ', np.mean(np.array(dicecoefs)))

print('average ious : ', np.mean(np.array(ious)))

print()



print('standard deviation of accuracy : ', np.std(np.array(accuracies)))

print('standard deviation of loss : ', np.std(np.array(losses)))

print('standard deviation of dicecoefs : ', np.std(np.array(dicecoefs)))

print('standard deviation of ious : ', np.std(np.array(ious)))



import pickle



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

        

    with open(str(h+1) + '_ultrasound_trainHistoryDict', 'wb') as file_pi:

        pickle.dump(history.history, file_pi)
selector = np.argmin(abs(np.array(ious) - np.mean(ious)))

model = load_model(str(selector+1)+'_unet_lung_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})
for i in range(10):

    index=np.random.randint(1,300)

    img = cv2.imread(train_files[index])

    img = cv2.resize(img, (512,512))

    img =img/255

    img = img[np.newaxis, :, :, :]

    pred = model.predict(img)



    plt.figure(figsize=(12,12))

    plt.subplot(1,3,1)

    plt.imshow(cv2.imread(train_files[index]))

    plt.title('Original Image')

    plt.subplot(1,3,2)

    plt.imshow(np.squeeze(cv2.imread(dilate_files[index])))

    plt.title('Original Mask')

    plt.subplot(1,3,3)

    plt.imshow(np.squeeze(pred) > .5)

    plt.title('Prediction')

    plt.show()