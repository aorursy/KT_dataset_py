import pandas as pd

import numpy as np

import os

import cv2



import shutil



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from skimage.io import imread, imshow

from skimage.transform import resize



# Don't Show Warning Messages

import warnings

warnings.filterwarnings('ignore')
# set up the canvas for the subplots

plt.figure(figsize=(7,7))

plt.tight_layout()

plt.axis('Off')





# Our subplot will contain 4 rows and 4 columns

# plt.subplot(nrows, ncols, plot_number)



# image

plt.subplot(1,2,1)

path = '../input/aisegmentcom-matting-human-datasets/clip_img/1803151818/clip_00000000/1803151818-00000003.jpg'

image = plt.imread(path)

plt.title('RGB Image')

plt.imshow(image)

plt.axis('off')





# image

plt.subplot(1,2,2)

path = '../input/aisegmentcom-matting-human-datasets/matting/1803151818/matting_00000000/1803151818-00000003.png'

mask = plt.imread(path)

plt.title('RGBA Image')

plt.imshow(mask)

plt.axis('off')



plt.show()
f = np.zeros((800,600,4))

f[:,:,:3] = image

f[:,:,3] = mask[:,:,3]

plt.imshow(f)
import seaborn as sns

sns.distplot(mask[:,:,3].flatten())
image.shape
(mask[:,:,3]).dtype
mask[:,:,3]
ana = list(mask[:,:,3].flatten())
len(ana)
ana.count(0)+ana.count(1)
plt.imshow(mask[:,:,3],cmap = "gray") # 0 opaque 1 transparent
dfio = pd.read_csv("../input/maincsv/dfio.csv")
dfio = pd.read_csv("../input/maincsv/dfio.csv")



for x in range(2) :

    

    path = dfio["input_path"][x]

    image = plt.imread(path)

    print(path)

    plt.title('RGB Image')

    plt.imshow(image)

    plt.axis('off')

    plt.show()





    

    path = dfio["output_path"][x]

    

    print(path)

    mask = plt.imread(path)

    plt.title('RGBA Image')

    plt.imshow(mask)

    plt.axis('off')



    plt.show()
df_v = dfio.iloc[:1280,:]

df_t = dfio.iloc[1280:23104,:]
print(len(df_v),len(df_t),len(dfio))
23148/64

361*64
df_t["output_path"][1280]
def train_generator( df,batch_size=64):

    IMG_HEIGHT = 128

    IMG_WIDTH = 128

    IMG_CHANNELS = 3



    

    while True:

        

        # load the data in chunks (batches)

        for  i in range(1280,18560,64):

           

            

            # Create empty X matrix - 3 channels

            # Note: We use len(df) because the last batch will be smaller than the other batches.

            X_train = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

            

            # create empty Y matrix - 1 channel

            Y_train = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)



            

            

            # Create X_train

            #================

            

            for x in range(64):

                

                # select the folder_id from the list

                



                # set the path to the image

                path = df["input_path"][x+i]

                # read the image

                image = cv2.imread(path)

                

                # resize the image

                image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

                p = image/255

                # insert the image into X_train

                X_train[x] = p

            

            

            # Create Y_train

            # ===============

                

           

                

                # select the folder_id from the list

               



                # set the path to the mask

                path = df["output_path"][x+i]



                # read the image

                mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)



                # select the alpha channel

                k = mask[:, :, 3]

                

                

                # expand dims from (800,600) to (800,600,1)

                k = np.expand_dims(k, axis=-1)

                

                # resize the mask

                k = resize(k, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

                k = k>0.5

                # insert the image into Y_train

                Y_train[x] = k



            yield X_train, Y_train
def val_generator( df1,batch_size=64):

    IMG_HEIGHT = 128

    IMG_WIDTH = 128

    IMG_CHANNELS = 3



    

    while True:

        

        # load the data in chunks (batches)

        for  i in range(0,len(df1),64):

           

            

            # Create empty X matrix - 3 channels

            # Note: We use len(df) because the last batch will be smaller than the other batches.

            X_train = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

            

            # create empty Y matrix - 1 channel

            Y_train = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)



            

            

            # Create X_train

            #================

            

            for x in range(64):

                

                # select the folder_id from the list

                



                # set the path to the image

                path = df1["input_path"][i+x]

                # read the image

                image = cv2.imread(path)

                

                # resize the image

                image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

                p = image/255

               

                # insert the image into X_train

                X_train[x] = p

            

            

            # Create Y_train

            # ===============

                

           

                

                # select the folder_id from the list

               



                # set the path to the mask

                path = df1["output_path"][x+i]



                # read the image

                mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)



                # select the alpha channel

                k = mask[:, :, 3]

                

                

                # expand dims from (800,600) to (800,600,1)

                k = np.expand_dims(k, axis=-1)

                

                # resize the mask

                k = resize(k, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

                k = k>0.5

                # insert the image into Y_train

                Y_train[x] = k



            yield X_train, Y_train
IMG_HEIGHT = 128

IMG_WIDTH = 128

IMG_CHANNELS = 3
from keras.models import Model, load_model

from keras.layers import Input, UpSampling2D

from keras.layers.core import Dropout, Lambda

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K



import tensorflow as tf
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))









c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)

c1 = Dropout(0.1) (c1)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)

p1 = MaxPooling2D((2, 2)) (c1)



c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)

c2 = Dropout(0.1) (c2)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)

p2 = MaxPooling2D((2, 2)) (c2)



c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)

c3 = Dropout(0.2) (c3)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)

p3 = MaxPooling2D((2, 2)) (c3)



c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)

c4 = Dropout(0.2) (c4)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)

p4 = MaxPooling2D(pool_size=(2, 2)) (c4)



c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)

c5 = Dropout(0.3) (c5)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)



u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)

u6 = concatenate([u6, c4])

c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)

c6 = Dropout(0.2) (c6)

c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)



u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)

u7 = concatenate([u7, c3])

c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)

c7 = Dropout(0.2) (c7)

c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)



u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)

u8 = concatenate([u8, c2])

c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)

c8 = Dropout(0.1) (c8)

c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)



u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)

u9 = concatenate([u9, c1], axis=3)

c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)

c9 = Dropout(0.1) (c9)

c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)



outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)



model = Model(inputs=[inputs], outputs=[outputs])



model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])



model.summary()
BATCH_SIZE = 64

num_train_samples = 17280

num_val_samples = len(df_v)

train_batch_size = BATCH_SIZE

val_batch_size = BATCH_SIZE



# determine numtrain steps

train_steps = 270

# determine num val steps

val_steps = np.ceil(num_val_samples / val_batch_size)
train_gen = train_generator(df_t,batch_size=BATCH_SIZE)

val_gen = val_generator(df_v,batch_size=BATCH_SIZE)
filepath = "model.h5"



earlystopper = EarlyStopping(patience=5, verbose=1,monitor="val_acc")



checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,  save_best_only=True, mode="auto")



                            

callbacks_list = [earlystopper, checkpoint]



history = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=10,validation_data=val_gen, validation_steps=val_steps,verbose=1,callbacks=callbacks_list)

                             

                            
epochs = 10
# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()