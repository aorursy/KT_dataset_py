# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import sys

import random

import warnings



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



from tqdm import tqdm

from itertools import chain

from skimage.io import imshow, imread_collection, concatenate_images, imread

from matplotlib.pyplot import imread

from skimage.transform import resize

from skimage.morphology import label



from tensorflow.python.keras.models import load_model

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint



from tensorflow.python.keras.layers import Input, Conv2D,Dropout,MaxPooling2D,Lambda,Conv2DTranspose

from tensorflow.python.keras import backend as K

from tensorflow.python.keras.layers.merge import concatenate as Concatenate

from tensorflow.python.keras import Model

import numpy as np

import tensorflow as tf

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Set some parameters

IMG_WIDTH = 128

IMG_HEIGHT = 128

IMG_CHANNELS = 3

TRAIN_PATH = '../input/stage1_train/'

TEST_PATH = '../input/stage2_test_final/'

# TEST_PATH = '../input/stage1_test/'



warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

seed = 42

random.seed = seed

np.random.seed = seed


# Get train and test IDs

train_ids = next(os.walk(TRAIN_PATH))[1] # os.walk 하위 디렉토리 검색

test_ids = next(os.walk(TEST_PATH))[1]

print("train length : ", len(train_ids))

print("test length : ", len(test_ids))

X_train=np.zeros((len(train_ids), IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)

Y_train=np.zeros((len(train_ids),IMG_HEIGHT,IMG_WIDTH,1), dtype=np.bool)



print('Getting and resizing train images and masks ... ')

sys.stdout.flush() 

for n,id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

    path=TRAIN_PATH+id_



    img=imread(path+"/images/"+id_+'.png')[:,:,:IMG_CHANNELS]

    img=resize(img, (IMG_HEIGHT,IMG_WIDTH), mode="constant",preserve_range=True)

    X_train[n]=img

    mask=np.zeros((IMG_HEIGHT,IMG_WIDTH,1),dtype=np.bool)

    for mask_file in next(os.walk(path+"/masks/"))[2]:

        mask_=imread(path+"/masks/"+mask_file)

        mask_=np.expand_dims(resize(mask_,(IMG_HEIGHT,IMG_WIDTH),mode="constant",preserve_range=True),axis=-1)

        mask=np.maximum(mask,mask_) # ?

    Y_train[n]=mask

# Get and resize test images

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

sizes_test = []

print('Getting and resizing test images ... ')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    path = TEST_PATH + id_

    print("n {}, id {}".format(n,id_))

    print(os.path.exists(path + '/images/' + id_ + '.png'))

    try :

        img = imread(path + '/images/' + id_ + '.png')

        if len(img.shape)!=3: continue

    except:

        continue

    img=img[:,:,:IMG_CHANNELS] # wrong png file

    

    print("n {}, id {} shape {}".format(n,id_,img.shape))



    #img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

    sizes_test.append([img.shape[0], img.shape[1]])

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_test[n] = img

    

print("X_test num : ",len(X_test))

print('Done!')
# Check if training data looks all right

ix = random.randint(0, len(train_ids))

imshow(X_train[ix])

plt.show()

imshow(np.squeeze(Y_train[ix]))

plt.show()
# Define IoU metric

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
def U_net(inputs):

    """

    

    :param inputs: train data

    :return:

    """

    input_shape=inputs[0].shape

    inputs=Input(input_shape)

    s=Lambda(lambda  x : x/255)(inputs)



    c1=Conv2D(16,(3,3), activation="elu", kernel_initializer="he_normal",padding="same")(s)

    c1=Dropout(0.1)(c1)

    c1=Conv2D(24,(3,3),activation='elu', kernel_initializer="he_normal",padding="same")(c1)

    p1=MaxPooling2D((2,2))(c1)



    c2=Conv2D(32,(3,3), activation="elu", kernel_initializer="he_normal",padding="same")(p1)

    c2=Dropout(0.1)(c2)

    c2=Conv2D(48,(3,3),activation='elu', kernel_initializer="he_normal",padding="same")(c2)

    p2=MaxPooling2D((2,2))(c2)



    c3=Conv2D(64,(3,3), activation="elu", kernel_initializer="he_normal",padding="same")(p2)

    c3=Dropout(0.1)(c3)

    c3=Conv2D(96,(3,3),activation='elu', kernel_initializer="he_normal",padding="same")(c3)

    p3=MaxPooling2D((2,2))(c3)



    c4=Conv2D(128,(3,3), activation="elu", kernel_initializer="he_normal",padding="same")(p3)

    c4=Dropout(0.1)(c4)

    c4=Conv2D(192,(3,3),activation='elu', kernel_initializer="he_normal",padding="same")(c4)

    p4=MaxPooling2D((2,2))(c4)



    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)

    c5 = Dropout(0.3)(c5)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)



    u6=Conv2DTranspose(128,(2,2),strides=(2,2),activation="elu", kernel_initializer="he_normal", padding="same")(c5)

    u6=Concatenate([u6,c4])

    c6=Conv2D(128,(3,3),activation="elu",kernel_initializer="he_normal",padding="same")(u6)

    c6=Dropout(0.2)(c6)

    c6=Conv2D(128, (3,3),activation="elu", kernel_initializer="he_normal",padding="same")(c6)



    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)

    u7 = Concatenate([u7, c3])

    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)

    c7 = Dropout(0.2)(c7)

    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)



    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)

    u8 = Concatenate([u8, c2])

    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)

    c8 = Dropout(0.1)(c8)

    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)



    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)

    u9 = Concatenate([u9, c1], axis=3)

    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)

    c9 = Dropout(0.1)(c9)

    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)



    outputs=Conv2D(1,(1,1), activation='sigmoid')(c9)



    model=Model(inputs, outputs)

    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[mean_iou])

    model.summary()



    return model
def U_net_pyramid(inputs):



    """

    piramid network 추가

    :param inputs: train data

    :return:

    """

    input_shape=inputs[0].shape

    inputs=Input(input_shape)

    s=Lambda(lambda  x : x/255)(inputs)



    c1=Conv2D(16,(3,3), activation="elu", kernel_initializer="he_normal",padding="same")(s)

    c1=Dropout(0.1)(c1)

    c1_s=Conv2D(8,(1,1),activation='elu', kernel_initializer="he_normal",padding="same")(c1) # pyramid

    c1_m=Conv2D(8,(3,3),activation='elu', kernel_initializer="he_normal",padding="same")(c1)  # pyramid

    c1_l = Conv2D(8, (5, 5), activation='elu', kernel_initializer="he_normal", padding="same")(c1)  # pyramid

    c1=Concatenate([c1_s,c1_m,c1_l])

    p1=MaxPooling2D((2,2))(c1)



    c2=Conv2D(32,(3,3), activation="elu", kernel_initializer="he_normal",padding="same")(p1)

    c2=Dropout(0.1)(c2)

    c2_s = Conv2D(16, (1,1), activation='elu', kernel_initializer="he_normal", padding="same")(c2)  # pyramid

    c2_m = Conv2D(16, (3,3), activation='elu', kernel_initializer="he_normal", padding="same")(c2)  # pyramid

    c2_l = Conv2D(16, (5, 5), activation='elu', kernel_initializer="he_normal", padding="same")(c2)   # pyramid

    c2 = Concatenate([c2_s, c2_m, c2_l])

    p2=MaxPooling2D((2,2))(c2)



    c3=Conv2D(64,(3,3), activation="elu", kernel_initializer="he_normal",padding="same")(p2)

    c3=Dropout(0.1)(c3)

    c3_s = Conv2D(32, (1,1), activation='elu', kernel_initializer="he_normal", padding="same")(c3)  # pyramid

    c3_m = Conv2D(32, (3,3), activation='elu', kernel_initializer="he_normal", padding="same")(c3)  # pyramid

    c3_l = Conv2D(32, (5, 5), activation='elu', kernel_initializer="he_normal", padding="same")(c3)  # pyramid

    c3 = Concatenate([c3_s, c3_m, c3_l])

    p3=MaxPooling2D((2,2))(c3)



    c4=Conv2D(128,(3,3), activation="elu", kernel_initializer="he_normal",padding="same")(p3)

    c4=Dropout(0.1)(c4)

    c4_s = Conv2D(64, (1,1), activation='elu', kernel_initializer="he_normal", padding="same")(c4)  # pyramid

    c4_m = Conv2D(64, (3,3), activation='elu', kernel_initializer="he_normal", padding="same")(c4)  # pyramid

    c4_l = Conv2D(64, (5, 5),activation='elu', kernel_initializer="he_normal", padding="same")(c4)  # pyramid

    c4 = Concatenate([c4_s, c4_m, c4_l])

    p4=MaxPooling2D((2,2))(c4)



    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)

    c5 = Dropout(0.3)(c5)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)



    u6=Conv2DTranspose(128,(2,2),strides=(2,2),activation="elu", kernel_initializer="he_normal", padding="same")(c5)

    u6=Concatenate([u6,c4])

    c6=Conv2D(128,(3,3),activation="elu",kernel_initializer="he_normal",padding="same")(u6)

    c6=Dropout(0.2)(c6)

    c6=Conv2D(128, (3,3),activation="elu", kernel_initializer="he_normal",padding="same")(c6)



    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)

    u7 = Concatenate([u7, c3])

    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)

    c7 = Dropout(0.2)(c7)

    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)



    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)

    u8 = Concatenate([u8, c2])

    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)

    c8 = Dropout(0.1)(c8)

    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)



    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)

    u9 = Concatenate([u9, c1], axis=3)

    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)

    c9 = Dropout(0.1)(c9)

    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)



    outputs=Conv2D(1,(1,1), activation='sigmoid')(c9)



    model=Model(inputs, outputs)

    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[mean_iou])

    model.summary()



    return model
# for train

def train(model):

    if model=='base':

        print("base model training ...")

        model=U_net(np.array(X_train))

        checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)

    else :

        print("pyramid model training ...")

        model=U_net_pyramid(np.array(X_train))

        checkpointer = ModelCheckpoint('model-dsbowl2018-1_pyramid.h5', verbose=1, save_best_only=True)



    earlystopper = EarlyStopping(patience=5, verbose=1)

    model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,

                        callbacks=[earlystopper, checkpointer])
def predict(X_train,X_test,model):

    if model=='base':

        print("base model load ...")

        model=load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})

    else :

        print("pyramid model load ...")

        model = load_model('model-dsbowl2018-1_pyramid.h5', custom_objects={'mean_iou': mean_iou})

    preds_train=model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)

    preds_val=model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

    preds_test=model.predict(X_test, verbose=1)



    # Threshold predictions

    preds_train_t = (preds_train > 0.5).astype(np.uint8)

    preds_val_t = (preds_val > 0.5).astype(np.uint8)

    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    preds_test_upsampled=[]

    for i in range(len(preds_test)):

        preds_test_upsampled.append(resize(np.squeeze([preds_test[i]]),(sizes_test[i][0], sizes_test[i][1]),

                                           mode="constant",preserve_range=True))



    

    print("visualization ...")

    # Perform a sanity check on some random training samples

    X_train=np.array(X_train)

    preds_val_t=np.array(preds_val_t)

    preds_train_t=np.array(preds_train_t)



    ix = random.randint(0, len(preds_train_t))

    # save image

    fig=plt.figure()

    ax=[]

    for i in range(3):

        ax.append(fig.add_subplot(1,3,i+1))



    ax[0].imshow(X_train[ix])

    ax[0].set_title("X_train")

    ax[1].imshow(np.squeeze(Y_train[ix]))

    ax[1].set_title("Y_train")

    ax[2].imshow(np.squeeze(preds_train_t[ix]))

    ax[2].set_title("X_pred")

    plt.show()

    #plt.savefig("./result/training_sample_{}.png".format(ix))





    # Perform a sanity check on some random validation samples

    ix = random.randint(0, len(preds_val_t))

    ix=36

    fig=plt.figure()

    ax=[]

    for i in range(3):

        ax.append(fig.add_subplot(1,3,i+1))



    ax[0].imshow(X_train[int(X_train.shape[0] * 0.9):][ix])

    ax[0].set_title("X_val")

    ax[1].imshow(np.squeeze(Y_train[int(Y_train.shape[0] * 0.9):][ix]))

    ax[1].set_title("Y_val")

    ax[2].imshow(np.squeeze(preds_val_t[ix]))

    ax[2].set_title("X_val_pred")

    plt.show()

    #plt.savefig("./result/validation_sample_{}.png".format(ix))

    return preds_test_upsampled
train("base")
train("pyramid")
predict(X_train,X_test, "base")
preds_test_upsampled=predict(X_train, X_test, "pyramid")
# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python

def rle_encoding(x):

    dots = np.where(x.T.flatten() == 1)[0]

    run_lengths = []

    prev = -2

    for b in dots:

        if (b>prev+1): run_lengths.extend((b + 1, 0))

        run_lengths[-1] += 1

        prev = b

    return run_lengths



def prob_to_rles(x, cutoff=0.5):

    lab_img = label(x > cutoff)

    for i in range(1, lab_img.max() + 1):

        yield rle_encoding(lab_img == i)
new_test_ids = []

rles = []

for n, id_ in enumerate(test_ids):

    rle = list(prob_to_rles(preds_test_upsampled[n]))

    rles.extend(rle)

    new_test_ids.extend([id_] * len(rle))
# Create submission DataFrame

sub = pd.DataFrame()

sub['ImageId'] = new_test_ids

sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

sub.to_csv('sub-dsbowl2018-1.csv', index=False)