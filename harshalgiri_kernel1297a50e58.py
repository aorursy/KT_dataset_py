import os

import random

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use("ggplot")

%matplotlib inline



import cv2

from tqdm import tqdm_notebook, tnrange

from glob import glob

from itertools import chain

from skimage.io import imread, imshow, concatenate_images

from skimage.transform import resize

from skimage.morphology import label

from sklearn.model_selection import train_test_split



import tensorflow as tf

from skimage.color import rgb2gray

from tensorflow.keras import Input

from tensorflow.keras.models import Model, load_model, save_model

from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate

from keras.layers.merge import concatenate

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers import Input,Concatenate, UpSampling2D



from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# set parameters

img_width = 112

img_height = 112

img_channels = 3
train_files = []

mask_files = glob('../input/lgg-mri-segmentation/kaggle_3m/*/*_mask*')



for i in mask_files:

    train_files.append(i.replace('_mask',''))



print(train_files[:10])

print(mask_files[:10])
#Lets plot some samples

rows,cols=4,4

fig=plt.figure(figsize=(10,10))

for i in range(1,rows*cols+1):

    fig.add_subplot(rows,cols,i)

    img_path=train_files[i]

    msk_path=mask_files[i]

    img=imread(img_path)

    msk=imread(msk_path)

    plt.imshow(img)

    plt.imshow(msk,alpha=0.4)

plt.show()
df = pd.DataFrame(data={"filename": train_files, 'mask' : mask_files})

df_train, df_test = train_test_split(df,test_size = 0.1)

df_train, df_val = train_test_split(df_train,test_size = 0.2)

print(df_train.shape)

print(df_val.shape)

print(df_test.shape)
def train_generator(data_frame, batch_size, aug_dict,

        image_color_mode="rgb",

        mask_color_mode="grayscale",

        image_save_prefix="image",

        mask_save_prefix="mask",

        save_to_dir=None,

        target_size=(256,256),

        seed=1):



    image_datagen = ImageDataGenerator(**aug_dict)

    mask_datagen = ImageDataGenerator(**aug_dict)

    

    image_generator = image_datagen.flow_from_dataframe(

        data_frame,

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

        x_col = "mask",

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
def contrastive_loss(y_true, y_pred):

    margin = 1

    return K.mean(y_true * K.square(y_pred) +

                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))



def iou(y_true, y_pred, smooth = 100):



    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)

    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection

    #sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)

    iou_acc = (intersection + smooth) / (union + smooth)

    return iou_acc

"""

def mean_iou(y_true, y_pred):

    prec = []

    for t in np.arange(0.5, 1.0, 0.05):

        y_pred_ = tf.to_int32(y_pred > t)

        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)

        K.get_session().run(tf.local_variables_initializer())

        with tf.control_dependencies([up_opt]):

            score = tf.identity(score)

        prec.append(score)

    return K.mean(K.stack(prec), axis=0)

"""

def dice_coef(y_true, y_pred):

    '''

    Params: y_true -- the labeled mask corresponding to an rgb image

            y_pred -- the predicted mask of an rgb image

    Returns: dice_coeff -- A metric that accounts for precision and recall

                           on the scale from 0 - 1. The closer to 1, the

                           better.

    Citation (MIT License): https://github.com/jocicmarko/

                            ultrasound-nerve-segmentation/blob/

                            master/train.py

    '''

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    smooth = 1.0

    return (2.0*intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)+smooth)



def precision(y_true, y_pred):

    

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def recall(y_true, y_pred):

    

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall





def dice_coef_loss(y_true, y_pred):

    return -dice_coef(y_true, y_pred)



def f1_score(y_true, y_pred):

    

    def recall(y_true, y_pred):

        

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def unet(input_size=(256,256,3)):

    inputs1 = Input(input_size)

    #s11 = Lambda(lambda x: x / 255) (inputs1)

    #s1 = Lambda(lambda x: x[:,:,:, 0:3])(s11)

    #s2 = Lambda(lambda x: x[:,:,:, 3:6])(s11)



    #inputs2 = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    #s2 = Lambda(lambda x: x / 255) (inputs2)



    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs1)

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)

    p1 = MaxPooling2D((2, 2)) (c1)



    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)

    p2 = MaxPooling2D((2, 2)) (c2)



    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)

    c3 = Dropout(0.2) (c3)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)

    p3 = MaxPooling2D((2, 2)) (c3)



    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)

    c4 = Dropout(0.2) (c4)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)

    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)



    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)

    c5 = Dropout(0.3) (c5)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)



    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)

    u6 = concatenate([u6, c4])

    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)

    c6 = Dropout(0.2) (c6)

    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)



    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)

    u7 = concatenate([u7, c3])

    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)

    c7 = Dropout(0.2) (c7)

    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)



    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)

    u8 = concatenate([u8, c2])

    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)

    c8 = Dropout(0.1) (c8)

    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)



    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)

    u9 = concatenate([u9, c1], axis=3)

    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)

    c9 = Dropout(0.1) (c9)

    c9_1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)



    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9_1)

    return Model(inputs=[inputs1], outputs=[outputs])

EPOCHS = 50

BATCH_SIZE = 32

learning_rate = 1e-2
train_generator_args = dict(rotation_range=0.2,

                            width_shift_range=0.5,

                            height_shift_range=0.5,

                            shear_range=0.05,

                            zoom_range=0.05,

                            horizontal_flip=True,

                            fill_mode='nearest')

train_gen = train_generator(df_train, BATCH_SIZE,

                                train_generator_args,

                                target_size=(img_height, img_width))

    

test_gener = train_generator(df_val, BATCH_SIZE,

                                dict(),

                                target_size=(img_height, img_width))

    

model = unet(input_size=(img_height, img_width, 3))







decay_rate = learning_rate / EPOCHS

opt = Adam(lr=learning_rate)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy',precision, recall,f1_score])



callbacks = [ModelCheckpoint('unet_brain_mri_seg.hdf5', verbose=1, save_best_only=True)]





history = model.fit(train_gen,

                    steps_per_epoch=len(df_train) / BATCH_SIZE, 

                    epochs=EPOCHS, 

                    callbacks=callbacks,

                    validation_data = test_gener,

                    validation_steps=len(df_val) / BATCH_SIZE)
a = history.history

#list_traindice = a['accuracy']

#list_testdice = a['val_accuracy']



list_trainjaccard = a['accuracy']

list_testjaccard = a['val_accuracy']



list_trainloss = a['f1_score']

list_testloss = a['val_f1_score']

plt.figure(1)

plt.plot(list_testloss, 'b-')

plt.plot(list_trainloss,'r-')

#plt.xlabel('iteration')

#plt.ylabel('loss')

plt.title('loss graph', fontsize = 15)

plt.figure(2)

plt.plot(list_traindice, 'r-')

plt.plot(list_testdice, 'b-')

plt.xlabel('iteration')

plt.ylabel('accuracy')

plt.title('accuracy graph', fontsize = 15)

plt.show()

#model = load_model('unet_brain_mri_seg.hdf5', custom_objects={'dice_coef': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})





test_gen = train_generator(df_test, BATCH_SIZE,

                                dict(),

                                target_size=(img_height, img_width))

results = model.evaluate(test_gen, steps=len(df_test) / BATCH_SIZE)

#print("Test lost: ",results[0])

print("Test IOU: ",results[1])

print("Test Dice Coefficent: ",results[2])
for i in range(30):

    index=np.random.randint(1,len(df_test.index))

    img = cv2.imread(df_test['filename'].iloc[index])

    img = cv2.resize(img ,(img_height, img_width))

    img = img / 255

    img = img[np.newaxis, :, :, :]

    pred=model.predict(img)



    plt.figure(figsize=(12,12))

    plt.subplot(1,3,1)

    plt.imshow(np.squeeze(img))

    plt.title('Original Image')

    plt.subplot(1,3,2)

    plt.imshow(np.squeeze(cv2.imread(df_test['mask'].iloc[index])))

    plt.title('Original Mask')

    plt.subplot(1,3,3)

    plt.imshow(np.squeeze(pred) > .5)

    plt.title('Prediction')

    plt.show()