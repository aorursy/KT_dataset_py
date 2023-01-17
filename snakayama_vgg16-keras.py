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
import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm

import random as rn

from sklearn.preprocessing import LabelEncoder



# dl libraries

from keras import backend as K

from keras.models import Sequential

from keras.models import Model

from keras.layers import Dense

from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop

from keras.utils import to_categorical

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping



 

# specifically for cnn

from keras.layers import Dropout, Flatten, Activation, Lambda

from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization



# preprocess

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf



import cv2

from sklearn.model_selection import train_test_split
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

train_image_path = '../input/train_images/'

test_image_path = '../input/test_images/'

df_submission = pd.read_csv('../input/sample_submission.csv')
df_train.shape
df_train['diagnosis'].hist(bins=4)
df_test.shape
df_submission.head(1)
len(os.listdir("../input/train_images/"))
len(os.listdir("../input/test_images/"))
train_images_name_list = os.listdir("../input/train_images/")

test_images_name_list = os.listdir("../input/test_images/")
sns.set_style("white")

count = 1

plt.figure(figsize=[20, 20])

for img_name in df_train['id_code'][:15]:

    img = cv2.imread("../input/train_images/%s.png" % img_name)[...,[2, 1, 0]]

    plt.subplot(5, 5, count)

    plt.imshow(img)

    plt.title("diagnosis %s" % df_train[df_train['id_code']==img_name]['diagnosis'])

    count += 1

    

plt.show()
X=[]

Z=[]

IMG_SIZE=150

ＴＲＡＩＮ_ＩＭＡＧＥ_DIR='../input/train_images/'



def assign_label(img, diagnosis):

    return diagnosis



def make_train_data(df_train, DIR):

    for img in tqdm(os.listdir(DIR)):

        #print(img)

        label = assign_label(img, df_train[df_train['id_code']==img.strip('.png')]['diagnosis'].values)

        path = os.path.join(DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)

        if not img is None:

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        else:

            print('None image file : ', path)

            continue



        X.append(np.array(img))

        Z.append(str(label))
make_train_data(df_train, ＴＲＡＩＮ_ＩＭＡＧＥ_DIR)

print("total images: ", len(X))

print("image size: ", X[0].shape)
fig, ax = plt.subplots(5,2)

fig.set_size_inches(15, 15)

for i in range(5):

    for j in range(2):

        l = rn.randint(0, len(Z))

        ax[i, j].imshow(X[l])

        ax[i, j].set_title('diagnosis: '+Z[l][1])



plt.tight_layout()
num_classes = 5

# Label encoding

le = LabelEncoder()

Y = le.fit_transform(Z)
Y.shape
# one hot vector

Y = to_categorical(Y)

# normalize

X = np.array(X)

X = X/255
Y[0]
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X)
_input = Input(shape=(150, 150, 3))

x = Lambda(lambda image: tf.image.resize_images(image, (224, 224)))(_input)

conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(conv1)

pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(conv2)



conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(pool1)

conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(conv3)

pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(conv4)



conv5 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(pool2)

conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(conv5)

conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(conv6)

pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(conv7)



conv8 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(pool3)

conv9 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(conv8)

conv10 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(conv9)

pool4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(conv10)



conv11 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(pool4)

conv12 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(conv11)

conv13 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(conv12)

pool5 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block5_pool')(conv13)



flat = Flatten(name='flatten')(pool5)

dense1 = Dense(4096, activation='relu', name='fc1')(flat)

dropout1 = Dropout(0.5, name='dropout1')(dense1)

dense2 = Dense(4096, activation='relu', name='fc2')(dropout1)

dropout2 = Dropout(0.5, name='dropout2')(dense2)

output = Dense(num_classes, activation='softmax', name='output')(dropout2)
from keras.callbacks import ReduceLROnPlateau

red_lr = ReduceLROnPlateau(monitor='val_acc',

                           factor=0.1,

                           patience=5,

                           verbose=1,

                           mode='auto',

                           min_delta=0.0001,

                           cooldown=0,

                           min_lr=0.00001)



earlyStopping = EarlyStopping(

    monitor='val_loss',

    patience=20,

    verbose=1

)



LOG_DIR = './logs'

if not os.path.isdir(LOG_DIR):

    os.mkdir(LOG_DIR)

else:

    pass

CKPT_PATH = LOG_DIR + '/checkpoint-{epoch:02d}-{val_loss:.4f}.hdf5'



tensorBoard = TensorBoard(

    log_dir=LOG_DIR,

    write_images=True

)



if not os.path.isdir('./logs/saved_medel'):

    os.mkdir('./logs/saved_medel')



mc = ModelCheckpoint('./logs/saved_medel/model.h5', monitor='val_loss', save_best_only = True, mode ='min', verbose = 1)
import keras

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)



model  = Model(inputs=_input, outputs=output)



model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])



model.summary()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
history = model.fit(x_train,y_train,validation_data=[x_test,y_test],batch_size=54,epochs=500,verbose=1,callbacks=[red_lr,earlyStopping,mc])
acc = history.history['acc']

val_acc = history.history['val_acc']

 

plt.plot(acc)

plt.plot(val_acc)

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper left')

plt.show()
loss = history.history['loss']

val_loss = history.history['val_loss']



plt.plot(loss)

plt.plot(val_loss)

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper left')

plt.show()
test_X=[]

test_Z=[]

IMG_SIZE=150

ＴEST_ＩＭＡＧＥ_DIR='../input/test_images/'



def make_test_data(DIR):

    for img in tqdm(os.listdir(DIR)):

        #print(img)

        #label = assign_label(img, df_train[df_train['id_code']==img.strip('.png')]['diagnosis'].values)

        path = os.path.join(DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)

        if not img is None:

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        else:

            print('None image file : ', path)

            continue



        test_X.append(np.array(img))

        #test_Z.append(str(label))
make_test_data(ＴEST_ＩＭＡＧＥ_DIR)
# normalize

test_X = np.array(test_X)

test_X = test_X/255
preds = model.predict(test_X)
preds.shape
preds_max = np.argmax(preds,axis = 1)
df_test['diagnosis']= preds_max
df_test['diagnosis'].hist(bins = 10)
df_test.to_csv('submission.csv',index=False)