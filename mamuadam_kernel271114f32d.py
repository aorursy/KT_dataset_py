import os

import json

import cv2

import keras

from keras import backend as K

from keras.models import Model

from keras.layers import Input

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.losses import binary_crossentropy

from keras.callbacks import Callback, ModelCheckpoint

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from keras.utils.vis_utils import plot_model

import random

#from segmentation_models import Unet

#from segmentation_models.backbones import get_preprocessing

random.seed(1)

from keras.models import load_model

from keras.preprocessing.image import img_to_array, load_img
train_df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

train_df.head()
mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()

print(mask_count_df.shape)

mask_count_df.head()
test_df = pd.read_csv('../input/severstal-steel-defect-detection/sample_submission.csv')

test_df.head()
cho=np.load("../input/chonpy/cho.npy")

cho=list(cho)
#ランダム幾何変換関数

def crop(A,B):

  B0=B[0]

  B1=B[1]

  B2=B[2]

  B3=B[3]

  if random.choice([0, 1]) == 0:

    A=cv2.flip(A,0)

    B0=cv2.flip(B0,0)

    B1=cv2.flip(B1,0)

    B2=cv2.flip(B2,0)

    B3=cv2.flip(B3,0)

  else:

    pass

  if random.choice([0,1])==0:

    A=cv2.flip(A,1)

    B0=cv2.flip(B0,1)

    B1=cv2.flip(B1,1)

    B2=cv2.flip(B2,1)

    B3=cv2.flip(B3,1)

  else:

    pass

  """

  rows,cols,aa = A.shape

  m1=random.choice(list(range(-5,5)))

  m2=random.choice(list(range(-5,5)))

  A = cv2.warpAffine(A,np.float32([[1,0,m1],[0,1,m2]]),(cols,rows))

  B0 = cv2.warpAffine(B0,np.float32([[1,0,m1],[0,1,m2]]),(cols,rows))

  B1 = cv2.warpAffine(B1,np.float32([[1,0,m1],[0,1,m2]]),(cols,rows))

  B2 = cv2.warpAffine(B2,np.float32([[1,0,m1],[0,1,m2]]),(cols,rows))

  B3 = cv2.warpAffine(B3,np.float32([[1,0,m1],[0,1,m2]]),(cols,rows))

  m3 = random.choice(list(range(-2,2)))

  A = cv2.warpAffine(A,cv2.getRotationMatrix2D((cols/2,rows/2),m3,1),(cols,rows))

  B0 = cv2.warpAffine(B0,cv2.getRotationMatrix2D((cols/2,rows/2),m3,1),(cols,rows))

  B1 = cv2.warpAffine(B1,cv2.getRotationMatrix2D((cols/2,rows/2),m3,1),(cols,rows))

  B2 = cv2.warpAffine(B2,cv2.getRotationMatrix2D((cols/2,rows/2),m3,1),(cols,rows))

  B3 = cv2.warpAffine(B3,cv2.getRotationMatrix2D((cols/2,rows/2),m3,1),(cols,rows))

  ra=random.uniform(-0.1, 0.1)

  img= np.zeros([rows,cols])

  if random.choice([0, 1]) == 0:

    A=np.reshape(A,(256,1600,1))

    img=np.reshape(img,(256,1600,1))

    for i in range(cols):

      weight = 1+((i*ra)/cols)

      img[:,i] = cv2.addWeighted(A[:,i],0.5*weight,A[:,i],0.5*weight,0)

    A=img

  else:

    img=img.T

    A=A.T

    A=np.reshape(A,(1600,256,1))

    img=np.reshape(img,(1600,256,1))

    for i in range(rows):

      weight = 1+((i*ra)/rows)

      img[:,i] = cv2.addWeighted(A[:,i],0.5*weight,A[:,i],0.5*weight,0)

    img=np.reshape(img,(256,1600))

    A=img.T

  """

  A=A*random.uniform(0.95, 1.05)

  B=[B0,B1,B2,B3]

  return(A,B)
def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def rle2mask(mask_rle, shape=(256,1600)):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (width,height) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T

def mask(gyo):

	train=train_df.iat[gyo,2].split(" ")

	train = [int(num) for num in train]

	mask = np.zeros(256*1600)

	mask=np.ravel(mask)



	for i in range(int(len(train)/2)):

		mask[train[2*i]:train[2*i]+train[2*i+1]-1]=[1]*(train[2*i+1]-1)

	mask=mask.reshape(1600,256)

	mask=mask.T

	mask.reshape(256,1600)

	mask=np.array(mask)

	return(train_df.iat[gyo,1],mask)
def build_masks(rles, input_shape):

    depth = len(rles)

    height, width = input_shape

    masks = np.zeros((height, width, depth))

    

    for i, rle in enumerate(rles):

        if type(rle) is str:

            masks[:, :, i] = rle2mask(rle, (width, height))

    

    return masks



def build_rles(masks):

    width, height, depth = masks.shape

    

    rles = [mask2rle(masks[:, :, i])

            for i in range(depth)]

    

    return rles
def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f

    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1. - score



def bce_dice_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
#data generator にて入出力を作る場合のコード

def batch_iter(data_size, batch_size):

    data_size=int((4/5)*data_size)

    num_batches_per_epoch = int(data_size / batch_size)



    def data_generator():

        while True:

            for batch_num in range(num_batches_per_epoch):

                start_index = batch_num * batch_size

                k=random.sample(list(range(data_size)), len(list(range(data_size))) )

                XX=[]

                YY=[]

                zeros=np.zeros([256,1600])

                for i in range(batch_size):

                	imageid=train_df.iat[k[i+start_index],0]



                	y=mask(k[i+start_index])

                	if k[i+start_index] in cho:

                	  x = cv2.imread("../input/train-images2/train_images2/"+str(k[i+start_index])+"_gyo.jpg", cv2.IMREAD_GRAYSCALE)                  

                	else:

                	  x = cv2.imread("../input/severstal-steel-defect-detection/train_images/"+imageid, cv2.IMREAD_GRAYSCALE)

                	x=np.array(x)

                	x=x/255

                	Y=[]

                	for j in range(4):

                		if y[0]==j+1:

                			Y.append(y[1])

                		else:

                			Y.append(zeros)

                	x,Y=crop(x,Y)

                	x=np.reshape(x, (256,1600,1))

                	Y=np.stack(Y,2)

                	XX.append(x)

                	YY.append(Y)

                X_train=np.array(XX)

                Y_train=np.array(YY)

                yield X_train, Y_train

    return num_batches_per_epoch, data_generator()

#4:1で学習

def batch_iter2(data_size, batch_size):

    num_batches_per_epoch = int(data_size / (5*batch_size))

    def data_generator2():

        while True:

            for batch_num in range(num_batches_per_epoch):

                start_index = int((4*data_size)/5+batch_num * batch_size)

                l0=[0]*(int((4*data_size)/5))

                l=random.sample(list(range(int((4*data_size)/5),data_size)), len(list(range(int((4*data_size)/5),data_size))) )

                l=l0+l

                XX=[]

                YY=[]

                zeros=np.zeros([256,1600])

                for i in range(batch_size):

                	imageid=train_df.iat[l[i+start_index],0]

                	y=mask(l[i+start_index])

                	if l[i+start_index] in cho:

                	  x = cv2.imread("../input/train-images2/train_images2/"+str(l[i+start_index])+"_gyo.jpg", cv2.IMREAD_GRAYSCALE)                  

                	else:

                	  x = cv2.imread("../input/severstal-steel-defect-detection/train_images/"+imageid, cv2.IMREAD_GRAYSCALE)

                	x=np.array(x)

                	x=x/255

                	Y=[]

                	for j in range(4):

                		if y[0]==j+1:

                			Y.append(y[1])

                		else:

                			Y.append(zeros)

                  

                	x,Y=crop(x,Y)

                	x=np.reshape(x, (256,1600,1))

                	Y=np.stack(Y,2)

                	XX.append(x)

                	YY.append(Y)

                X_train=np.array(XX)

                Y_train=np.array(YY)

                yield X_train, Y_train

    return num_batches_per_epoch, data_generator2()
#履歴

def plot_history(history, outdir):

    # 精度の履歴をプロット

    plt.figure()

    plt.plot(history.history['dice_coef'], marker='.')

    plt.plot(history.history['val_dice_coef'], marker='.')

    plt.title('model dice_coef')

    plt.xlabel('epoch')

    plt.ylabel('dice_coef')

    plt.grid()

    plt.legend(['train', 'test'], loc='upper left')

###

#    plt.savefig(os.path.join(outdir, 'dice_coef5.png'))

    # 損失の履歴をプロット

    plt.figure()

    plt.plot(history.history['loss'], marker='.')

    plt.plot(history.history['val_loss'], marker='.')

    plt.title('model loss')

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.grid()

    plt.legend(['train', 'test'], loc='upper left')

###保存

#    plt.savefig(os.path.join(outdir, 'loss5.png'))
#model

def build_model(input_shape):

    inputs = Input(input_shape)



    c1 = Conv2D(8, (3, 3), activation='elu', padding='same') (inputs)

    c1 = Conv2D(8, (3, 3), activation='elu', padding='same') (c1)

    p1 = MaxPooling2D((2, 2)) (c1)



    c2 = Conv2D(16, (3, 3), activation='elu', padding='same') (p1)

    c2 = Conv2D(16, (3, 3), activation='elu', padding='same') (c2)

    p2 = MaxPooling2D((2, 2)) (c2)



    c3 = Conv2D(32, (3, 3), activation='elu', padding='same') (p2)

    c3 = Conv2D(32, (3, 3), activation='elu', padding='same') (c3)

    p3 = MaxPooling2D((2, 2)) (c3)



    c4 = Conv2D(64, (3, 3), activation='elu', padding='same') (p3)

    c4 = Conv2D(64, (3, 3), activation='elu', padding='same') (c4)

    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)



    c5 = Conv2D(64, (3, 3), activation='elu', padding='same') (p4)

    c5 = Conv2D(64, (3, 3), activation='elu', padding='same') (c5)

    p5 = MaxPooling2D(pool_size=(2, 2)) (c5)



    c55 = Conv2D(128, (3, 3), activation='elu', padding='same') (p5)

    c55 = Conv2D(128, (3, 3), activation='elu', padding='same') (c55)



    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c55)

    u6 = concatenate([u6, c5])

    c6 = Conv2D(64, (3, 3), activation='elu', padding='same') (u6)

    c6 = Conv2D(64, (3, 3), activation='elu', padding='same') (c6)



    u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)

    u71 = concatenate([u71, c4])

    c71 = Conv2D(32, (3, 3), activation='elu', padding='same') (u71)

    c61 = Conv2D(32, (3, 3), activation='elu', padding='same') (c71)



    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c61)

    u7 = concatenate([u7, c3])

    c7 = Conv2D(32, (3, 3), activation='elu', padding='same') (u7)

    c7 = Conv2D(32, (3, 3), activation='elu', padding='same') (c7)



    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)

    u8 = concatenate([u8, c2])

    c8 = Conv2D(16, (3, 3), activation='elu', padding='same') (u8)

    c8 = Conv2D(16, (3, 3), activation='elu', padding='same') (c8)



    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)

    u9 = concatenate([u9, c1], axis=3)

    c9 = Conv2D(8, (3, 3), activation='elu', padding='same') (u9)

    c9 = Conv2D(8, (3, 3), activation='elu', padding='same') (c9)



    outputs = Conv2D(4, (1, 1), activation='sigmoid') (c9)



    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])

    

    return model



model = build_model((256, 1600, 1))
checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor='val_dice_coef', 

    verbose=0, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)
#学習開始

#history=model.fit_generator(batch_iter(len(train_df)-len(train_df)%128, 16)[1], batch_iter(len(train_df)-len(train_df)%128, 16)[0],validation_data=batch_iter2(len(train_df)-len(train_df)%128,4)[1],validation_steps=batch_iter2(len(train_df)-len(train_df)%128,4)[0],callbacks=[checkpoint], epochs=30,verbose=1)
"""

with open('history.json', 'w') as f:

    json.dump(history.history, f)



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['dice_coef', 'val_dice_coef']].plot()

"""
"""

imageid=[]

for i in range(len(test_df)):

  for j in range(4):

    imageid.append(test_df.iat[i,0])

encode=['']*len(imageid)

classid=[]

for i in range(len(test_df)):

  for j in range(4):

    classid.append(j+1)

sample_submission=pd.DataFrame({'ImageId':imageid,'EncodedPixels':encode,'ClassId':classid})

"""


imageid=[]

for i in range(len(test_df)):

  for j in range(4):

    imageid.append(test_df.iat[i,0]+"_"+str(j+1))

encode=['1 1']*len(imageid)

sample_submission=pd.DataFrame({'ImageId_ClassId':imageid,'EncodedPixels':encode})



model.load_weights('../input/weights3/weights.3')

for i in range(len(test_df)):

#for i in range(200):

  img = cv2.imread("../input/severstal-steel-defect-detection/test_images/"+test_df.iat[i,0], cv2.IMREAD_GRAYSCALE)

  X=[]

  img=img.reshape(256,1600,1)

  img = np.array(img)/255

  X.append(img)

  X=np.array(X)

  pred = model.predict(X, batch_size=1, verbose=0)

  C=pred[0]

  C=np.where(C >0.5, 1, 0)

  C=build_rles(C)

  for j in range(4):

    sample_submission.iat[4*i+j,1]=C[j]





"""

indexNames = sample_submission[ sample_submission['EncodedPixels'] == '' ].index

# Delete these row indexes from dataFrame

sample_submission.drop(indexNames , inplace=True)

"""



"""

sample_submission = pd.concat([test_df, sample_submission])

indexNames = sample_submission[ sample_submission['EncodedPixels'] == '' ].index

sample_submission.drop(indexNames , inplace=True)

"""
"""

print(sample_submission)

"""
"""

sample_submission=pd.read_csv('../input/severstal-steel-defect-detection/sample_submission.csv')

print(sample_submission.iat[0,0])

"""


sample_submission.to_csv('submission.csv', index=False)