import numpy as np 

import pandas as pd 

import os

import glob

import tensorflow as tf

import skimage

from PIL import Image

from tensorflow import keras

from tensorflow.keras.models import *

from tensorflow.keras.layers import *

from tensorflow.keras.optimizers import *

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger

import matplotlib.pyplot as plt

from skimage.exposure import equalize_adapthist

from cv2 import equalizeHist
IMAGES_PATH = "/kaggle/input/saving-infected-images-and-masks-as-jpg/train/"

MASKS_PATH = "/kaggle/input/saving-infected-images-and-masks-as-jpg/train/"



# IMAGE_SIZE = (224, 224)

IMAGE_SIZE = (512, 512)

CLASS_MODE = None

COLOR_MODE = 'grayscale'

EPOCHS = 50

BATCH_SIZE = 8

# BATCH_SIZE = 1

SEED = 1337
os.listdir(IMAGES_PATH)
test = glob.glob(f'{IMAGES_PATH}/images/*.jpg')

test_mask_path = glob.glob(f'{MASKS_PATH}/masks/*.jpg')
# from skimage import io

test_img = np.array(Image.open(test[0]).convert('L'))

test_mask = np.array(Image.open(test_mask_path[0]).convert('L'))

# test_img = io.imread(test[0])

# test_mask = io.imread(test_mask_path[0])



plt.imshow(test_img, cmap='gray')

# plt.imshow(test_img + test_mask * 0.1, cmap='gray')

# plt.imshow(test_mask, cmap='gray')
plt.imshow(test_img.astype(np.uint8), cmap='gray')
equalized_test_img = equalize_adapthist(np.array(test_img) * 1./255)

plt.imshow(equalized_test_img, cmap='gray')
# equalized_test_img = test_img * 1./255

equalized_test_img = equalizeHist(test_img)

plt.imshow(equalized_test_img,  cmap='gray')

print(equalized_test_img.shape)
def apply_adaptive_histogram_equalization(img):

    img = equalizeHist(img.astype(np.uint8))

    img = np.expand_dims(img.astype(np.float32), -1)

    return img
from tensorflow.keras.preprocessing.image import ImageDataGenerator





# VI Note: use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same

data_gen_args = dict(rescale=1./255, 

                     preprocessing_function=apply_adaptive_histogram_equalization,

                     rotation_range=10,

                     shear_range=0.2,

                    width_shift_range=0.01,

                    horizontal_flip=True,

                     validation_split=0.2)



# So our usage here is as data loader instead of loading everything in RAM, not data augmentation

mask_gen_args = dict(rescale=1./255,

                    rotation_range=10,

                     shear_range=0.2,

                    width_shift_range=0.01,

                    horizontal_flip=True,

                    validation_split=0.2)  # to make it binary, add: preprocessing_function=apply_adaptive_equalization,



image_datagen = ImageDataGenerator(**data_gen_args)

mask_datagen  = ImageDataGenerator(**mask_gen_args) 



# Provide the same seed and keyword arguments to the fit and flow methods



image_generator = image_datagen.flow_from_directory(

    IMAGES_PATH,

    class_mode=CLASS_MODE,

    classes=['images'],

    seed=SEED,

    batch_size=BATCH_SIZE,

    color_mode=COLOR_MODE,

    target_size=IMAGE_SIZE,

    subset='training'

)



mask_generator = mask_datagen.flow_from_directory(

    MASKS_PATH,

    classes=['masks'],

    class_mode=CLASS_MODE,

    seed=SEED,

    batch_size=BATCH_SIZE,

    color_mode=COLOR_MODE,

    target_size=IMAGE_SIZE,

    subset='training'

)



# combine generators into one which yields image and masks

train_generator = zip(image_generator, mask_generator)



      



val_image_generator = image_datagen.flow_from_directory(

    IMAGES_PATH,

    class_mode=CLASS_MODE,

    classes=['images'],

    color_mode=COLOR_MODE,

    seed=SEED,

    batch_size=BATCH_SIZE,

    target_size=IMAGE_SIZE,

    subset='validation'

)



val_mask_generator = mask_datagen.flow_from_directory(

    MASKS_PATH,

    classes=['masks'],

    class_mode=CLASS_MODE,

    seed=SEED,

    batch_size=BATCH_SIZE,

    color_mode=COLOR_MODE,

    target_size=IMAGE_SIZE,

    subset='validation'

)



# combine generators into one which yields image and masks

val_generator = zip(val_image_generator, val_mask_generator)


def train_generator_fn():

    for (img,mask) in train_generator:

        yield (img, mask)        





def val_generator_fn():

    for (img,mask) in val_generator:

        yield (img, mask)  

        

        

print(f"Number of training examples: {len(os.listdir(os.path.join(IMAGES_PATH, 'images')))}")
(x, m) = next(train_generator_fn())

# plt.imshow(np.squeeze(x[1]), cmap='gray')

plt.imshow(np.squeeze(x[1]) + np.squeeze(m[1]) * 0.5, cmap='gray')
plt.imshow(np.squeeze(m[1]), cmap='gray')
# Evaluation metric for the competition.

from keras.losses import binary_crossentropy



def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = tf.keras.layers.Flatten()(y_true)

    y_pred_f = tf.keras.layers.Flatten()(y_pred)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)





def dice_coef_loss(y_true, y_pred):

    return 1.0 - dice_coef(y_true, y_pred)



def bce_dice_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)
# https://www.kaggle.com/cpmpml/fast-iou-metric-in-numpy-and-tensorflow

from tensorflow.keras import backend as K

def get_iou_vector(A, B):

    # Numpy version

    B = K.cast(B, 'float32')

    batch_size = A.shape[0]

    if batch_size is None:

      batch_size = 0

    metric = 0.0

    for batch in range(batch_size):

        t, p = A[batch], B[batch]

        true = np.sum(t)

        pred = np.sum(p)



        # deal with empty mask first

        if true == 0:

            pred_batch_size = pred / ( p.shape[0] * p.shape[1] )

            if pred_batch_size > 0.03:

               pred_batch_size = 1 

            metric +=  1 - pred_batch_size

            continue

        

        # non empty mask case.  Union is never empty 

        # hence it is safe to divide by its number of pixels



        intersection = np.sum(t * p)

        union = true + pred - intersection

        iou = intersection / union

        

        # iou metrric is a stepwise approximation of the real iou over 0.5

        iou = np.floor(max(0, (iou - 0.45)*20)) / 10

        

        metric += iou

        

    # teake the average over all images in batch

    metric /= batch_size

    return metric





def my_iou_metric(label, pred):

    # Tensorflow version

    return tf.py_function(get_iou_vector, [label, pred > 0.5], tf.float64)

factor = 0.5

lr_patience = 3

lr_cooldown = 1

learning_rate = 1e-4

loss_function = bce_dice_loss
def create_dir(dirname):

    try:

        os.makedirs(dirname)

        print(f"Directory '{dirname}' created.") 

    except FileExistsError:

        print(f"Directory '{dirname}' already exists.")
models_dir = '/kaggle/working/models/'

create_dir(models_dir)



model_name = f'M-epochs-{EPOCHS}-lr-{learning_rate}-reduce-{factor}-each-{lr_patience}-loss-{loss_function.__name__}'

model_path = os.path.join(models_dir, model_name)

best_model_path = os.path.join(model_path, 'best')

model_epochs_path = os.path.join(model_path, 'epochs')

model_logs_path = os.path.join(model_path, 'logs')



create_dir(best_model_path)

create_dir(model_epochs_path)

create_dir(model_logs_path)
# def unet(n_classes, input_size = (*IMAGE_SIZE, 1), flat=False):

#     inputs = Input(input_size)

#     conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)

#     conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    

#     conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)

#     conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)

#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    

#     conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)

#     conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    

#     conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)

#     conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

#     drop4 = Dropout(0.5)(conv4)

#     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)



#     conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)

#     conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

#     drop5 = Dropout(0.5)(conv5)



#     up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))

#     merge6 = concatenate([drop4,up6], axis = 3)

#     conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)

#     conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)



#     up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))

#     merge7 = concatenate([conv3,up7], axis = 3)

#     conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)

#     conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)



#     up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))

#     merge8 = concatenate([conv2,up8], axis = 3)

#     conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)

#     conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)



#     up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))

#     merge9 = concatenate([conv1,up9], axis = 3)

#     conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)

#     conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

#     #conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

#     #conv10 = Conv2D(n_classes, (1,1), activation = 'softmax')(conv9)

#     conv10 = Conv2D(n_classes, (1,1), padding='same')(conv9)

# #     if flat:

# #       output_layer = Reshape((256*256,n_classes))(conv10)

# #     else:

#     output_layer = conv10

#     output_layer = Activation('sigmoid')(output_layer)

     

#     model = Model(inputs,output_layer)





#     return model
def unet(n_classes, input_size = (*IMAGE_SIZE, 1), flat=False):

    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    

    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)

    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

    drop4 = Dropout(0.5)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    

    conv44 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)

    conv44 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv44)

    drop44 = Dropout(0.5)(conv44)

    pool44 = MaxPooling2D(pool_size=(2, 2))(drop44)



    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool44)

    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    drop5 = Dropout(0.5)(conv5)

    

    up66 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))

    merge66to4 = concatenate([drop44,up66], axis = 3)

    conv66 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge66to4)

    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv66)



    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv66))

    merge6to4 = concatenate([drop4,up6], axis = 3)

    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6to4)

    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)



    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))

    merge7to3 = concatenate([conv3,up7], axis = 3)

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7to3)

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)



    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))

    merge8to2 = concatenate([conv2,up8], axis = 3)

    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8to2)

    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)



    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))

    merge9to1 = concatenate([conv1,up9], axis = 3)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9to1)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    

    

    conv10 = Conv2D(n_classes, (1,1), padding='same')(conv9)

    output_layer = conv10

    output_layer = Activation('sigmoid')(output_layer)

     

    model = Model(inputs,output_layer)





    return model
model = unet(n_classes=1)  # n_classes=1 not 2 -> see last layer output



model.summary()
from tensorflow.keras.metrics import MeanIoU



# Conclusion: Don't use dice_coef_loss as a loss function in its own.

# very high starting LR -> bad results

# model.compile(optimizer = Adam(0.1), loss=dice_coef_loss, metrics=[dice_coef, 'accuracy', MeanIoU(num_classes=2)])

# model.compile(optimizer = Adam(0.01), loss=dice_coef_loss, metrics=[dice_coef, 'accuracy', MeanIoU(num_classes=2)])

# model.compile(optimizer = Adam(lr = 0.0001), loss=dice_coef_loss, metrics=[dice_coef, 'accuracy', MeanIoU(num_classes=2)])

# 





# Good starting loss

# model.compile(optimizer = Adam(lr = 1e-5), loss=BinaryCrossentropy(), metrics=[dice_coef, 'accuracy', MeanIoU(num_classes=2)])

# ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Epoch 00050: val_loss did not improve from 0.03727

# 267/267 [==============================] - 79s 297ms/step - loss: 0.0280 - dice_coef: 0.2126 - accuracy: 0.9859 - mean_io_u_1: 0.4959 - val_loss: 0.0514 - val_dice_coef: 0.1394 - val_accuracy: 0.9808 - val_mean_io_u_1: 0.4963

# Best: 

# at epoch 11:

# 0.9856 - mean_io_u_1: 0.4959 - val_loss: 0.0373 - val_dice_coef: 0.0971 - val_accuracy: 0.9858 - val_mean_io_u_1: 0.4963









# ---------------------------------------

# learning_rate = 1e-3

# model.compile(optimizer = Adam(lr = learning_rate), loss=BinaryCrossentropy(), metrics=[dice_coef, 'accuracy', MeanIoU(num_classes=2)])

# ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6, verbose=1)



# Epoch 00050: val_loss did not improve from 0.03023

# 267/267 [==============================] - 79s 298ms/step - loss: 0.0360 - dice_coef: 0.1134 - accuracy: 0.9850 - mean_io_u_2: 0.4958 - val_loss: 0.0357 - val_dice_coef: 0.1118 - val_accuracy: 0.9856 - val_mean_io_u_2: 0.4964

# Best:  

# Epoch 00046: val_loss improved from 0.03032 to 0.03023, saving model to models/best/best_pneumorhorax_dice.h5

# 267/267 [==============================] - 80s 300ms/step - loss: 0.0371 - dice_coef: 0.1073 - accuracy: 0.9852 - mean_io_u_2: 0.4957 - val_loss: 0.0302 - val_dice_coef: 0.1034 - val_accuracy: 0.9864 - val_mean_io_u_2: 0.4966

# Best second training

# Epoch 00070: val_loss improved from 0.02977 to 0.02966, saving model to models/best/best_pneumorhorax_dice.h5

# 267/267 [==============================] - 83s 311ms/step - loss: 0.0248 - dice_coef: 0.2727 - accuracy: 0.9870 - mean_io_u_2: 0.4960 - val_loss: 0.0297 - val_dice_coef: 0.1973 - val_accuracy: 0.9866 - val_mean_io_u_2: 0.4966

# last in second

# Epoch 00100: ReduceLROnPlateau reducing learning rate to 1.5625000742147677e-05.

# 267/267 [==============================] - 82s 305ms/step - loss: 0.0108 - dice_coef: 0.6561 - accuracy: 0.9906 - mean_io_u_2: 0.5046 - val_loss: 0.0446 - val_dice_coef: 0.2282 - val_accuracy: 0.9848 - val_mean_io_u_2: 0.4972

# model.compile(optimizer = Adam(lr = 1e-4), loss=BinaryCrossentropy(), metrics=[dice_coef, 'accuracy', MeanIoU(num_classes=2)])

# ---------------------------------------

# Conclusion: start with loss 1e-4 or 1e-5

# Patience of 10 is very big, try 5, maybe you should use cooldown=1, becuase after reducing lr the val get small

# ---------------------------------------



# best learning rate to start with 1e-4

model.compile(optimizer = Adam(lr = learning_rate), loss=loss_function, metrics=[dice_coef, 'accuracy', MeanIoU(num_classes=2), my_iou_metric])





# model.compile(optimizer = Adam(lr = 1e-5), loss=BinaryCrossentropy(), metrics=[dice_coef, 'accuracy', MeanIoU(num_classes=2)])

# model.compile(optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
print(model_name)



callbacks = [

    ModelCheckpoint(os.path.join(best_model_path, 'best_pneumorhorax_dice.h5'), monitor='val_loss',verbose=1, save_best_only=True),

    ModelCheckpoint(filepath=os.path.join(model_epochs_path, 'model.{epoch:02d}-{val_loss:.2f}.h5'), save_freq='epoch', period=10),

    ReduceLROnPlateau(factor=factor, patience=lr_patience, min_lr=1e-6, verbose=1, cooldown=lr_cooldown),

    TensorBoard(model_logs_path), 

    CSVLogger(os.path.join(model_path, "model_history_log.csv"), append=True)

]
# history = model.fit_generator(train_generator_fn(),

#                     validation_data=val_generator_fn(),

#                     steps_per_epoch=len(image_generator),

#                     validation_steps=len(val_image_generator),

#                     epochs=EPOCHS,

#                     callbacks=callbacks)


pretrained_model_path = "/kaggle/input/pneumorhorax-model/epoch33best_pneumorhorax_dice.h5"

pretrained_model = tf.keras.models.load_model(pretrained_model_path,

                                       custom_objects={

                                           'dice_coef': dice_coef,

                                           'bce_dice_loss': bce_dice_loss

                                       }

                                      )

pretrained_model.compile(optimizer = Adam(lr = 2e-5), loss=bce_dice_loss, metrics=[dice_coef, 'accuracy', MeanIoU(num_classes=2), my_iou_metric])
history = pretrained_model.fit_generator(train_generator_fn(),

                    validation_data=val_generator_fn(),

                    steps_per_epoch=len(image_generator),

                    validation_steps=len(val_image_generator),

                    epochs=EPOCHS,

                    callbacks=callbacks)



# 267/267 [==============================] - 333s 1s/step - loss: 0.7828 - dice_coef: 0.2802 - accuracy: 0.9793 - mean_io_u_4: 0.4965 - my_iou_metric: 0.0228 - val_loss: 0.7841 - val_dice_coef: 0.2771 - val_accuracy: 0.9759 - val_mean_io_u_4: 0.4969 - val_my_iou_metric: 0.0160
def plot_learning_metrics(history_model):



    plt.plot(history_model.history['loss'], label='loss')

    plt.plot(history_model.history['val_loss'], label = 'val_loss')



    plt.xlabel('Epoch')

    plt.ylabel('Loss')

#     plt.ylim([0.5, 1])

    plt.legend(loc='lower right')



    plt.show()
def plot_dice_history(history_model):



    plt.plot(history_model.history['dice_coef'], label='dice_coef')

    plt.plot(history_model.history['val_dice_coef'], label = 'val_dice_coef')

    plt.xlabel('Epoch')

    plt.ylabel('Dice Coef')

#     plt.ylim([0.5, 1])

    plt.legend(loc='lower right')



    plt.show()
plot_learning_metrics(history)
plot_dice_history(history)
import pickle 

try: 

    os.makedirs("/kaggle/working/models/history/")

except FileExistsError: 

    print("exists")



with open("/kaggle/working/models/history/train_history_dict", 'wb') as file_pi:

        pickle.dump(history.history, file_pi)
import glob

test_images_paths = glob.glob("/kaggle/input/siim-acr-pneumothorax-segmentation/stage_2_images/*.dcm")

test_images_paths[:5]
def get_image_name_from_path(image_path):

    image_name_with_extesion = image_path.rsplit('/', maxsplit=1)[-1]

    image_name_without_extension = image_name_with_extesion.split('.')[0]

    return image_name_without_extension



# get_image_name_from_path('/kaggle/input/siim-acr-pneumothorax-segmentation/stage_2_images/ID_9979c1b39.dcm')
test_images_names = list(map(get_image_name_from_path, test_images_paths))

test_images_names[:5]
import pandas as pd 



test_df = pd.DataFrame(dict(ImageId=test_images_names,

               ImagePath=test_images_paths,

               EncodedPixels=-1))



test_df.head()
import numpy as np

import pandas as pd

import pydicom

import cv2

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras_preprocessing.image.dataframe_iterator import DataFrameIterator







class DCMDataFrameIterator(DataFrameIterator):

    def __init__(self, *arg, **kwargs):

        # add dcm as a valid file

        white_list_formats_with_dcm = list(self.white_list_formats) + ['dcm']

        self.white_list_formats = tuple(white_list_formats_with_dcm) 

        

        super(DCMDataFrameIterator, self).__init__(*arg, **kwargs)

        

        self.dataframe = kwargs['dataframe']

        self.x = self.dataframe[kwargs['x_col']]

        self.y = self.dataframe[kwargs['y_col']]

        self.color_mode = kwargs['color_mode']

        self.target_size = kwargs['target_size']



    def _get_batches_of_transformed_samples(self, indices_array):

        # get batch of images

        batch_x = np.array([self.read_dcm_as_array(dcm_path, self.target_size, color_mode=self.color_mode)

                            for dcm_path in self.x.iloc[indices_array]])



        batch_y = np.array(self.y.iloc[indices_array].astype(np.uint8))  # astype because y was passed as str



        # transform images

        if self.image_data_generator is not None:

            for i, (x, y) in enumerate(zip(batch_x, batch_y)):

                transform_params = self.image_data_generator.get_random_transform(x.shape)

                batch_x[i] = self.image_data_generator.apply_transform(x, transform_params)

                # you can change y here as well, eg: in semantic segmentation you want to transform masks as well 

                # using the same image_data_generator transformations.



        return batch_x, batch_y



    @staticmethod

    def read_dcm_as_array(dcm_path, target_size=(256, 256), color_mode='rgb'):

        image_array = pydicom.dcmread(dcm_path).pixel_array

        image_array = cv2.resize(image_array, target_size, interpolation=cv2.INTER_NEAREST)  #this returns a 2d array

        image_array = np.expand_dims(image_array, -1)

        if color_mode == 'rgb':

            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

        return image_array
test_datagen = ImageDataGenerator(rescale=1./255)



# Using the testing generator to evaluate the model after training

test_augmentation_parameters = dict(

    rescale=1.0/255.0

)



test_consts = {

    'batch_size': 1,  # should be 1 in testing

    'class_mode': CLASS_MODE,

    'color_mode': COLOR_MODE,

    'target_size': IMAGE_SIZE,  # resize input images

    'shuffle': False

}



test_augmenter = ImageDataGenerator(**test_augmentation_parameters)



test_generator = DCMDataFrameIterator(dataframe=test_df,

                             x_col='image_path',

                             y_col='target',

                             image_data_generator=None,

                             **test_consts)



# new_model = tf.keras.models.load_model('models/interrupted_model.h5',

#                                        custom_objects={

#                                            'dice_coef': dice_coef,

#                                            'dice_coef_loss': dice_coef_loss

#                                        }

#                                       )



new_model = tf.keras.models.load_model('models/interrupted_model.h5',

                                       compile=False)



# Check its architecture

new_model.summary()
predict = model.predict_generator(test_generator)