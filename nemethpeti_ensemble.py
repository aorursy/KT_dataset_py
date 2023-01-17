!mkdir efficientnet

!cp -r ../input/segmentation-masters/efficientnet-master/efficientnet-master/efficientnet/* ./efficientnet



!mkdir classification_models

!cp -r ../input/segmentation-masters/classification_models-master/classification_models-master/classification_models/* ./classification_models



!mkdir segmentation_models

!cp -r ../input/segmentation-masters/segmentation_models-master/segmentation_models-master/segmentation_models/* ./segmentation_models
import os



import matplotlib.pyplot as plt

import random

import math as math

import numpy as np

import pandas as pd

import cv2

import os

import tensorflow as tf

import keras.backend as K

from keras.utils import Sequence

from keras import Model

from keras.losses import binary_crossentropy

from keras.optimizers import Adam



import segmentation_models as sm



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

 



def rle2mask(mask_rle, shape=(2100, 1400)):



    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    

    if mask_rle != None and type(mask_rle) is str: 

        s = mask_rle.split()



        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

        starts -= 1

        ends = starts + lengths



        for lo, hi in zip(starts, ends):

            img[lo:hi] = 1

            

    return img.reshape(shape).T



def normalize(images):

    return images/128-1

    

def denormalize(images):

    return ((images+1)*128).astype('uint8')



def load_image(Image):

    path = TEST_PATH + Image

    #print(path)

    

    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if (img.shape != (height, width, 3)):

        img = cv2.resize(img, (width, height))    



    return img



def resizeMask(mask, w, h):

    

    resmask = np.zeros((h, w, mask.shape[2]))

    for i in range(mask.shape[2]):

        resmask[...,i] = cv2.resize(mask[...,i], (w,h))

        

    return resmask

TEST_PATH = '../input/understanding_cloud_organization/test_images/'

test_df = pd.read_csv('../input/understanding_cloud_organization/sample_submission.csv')

test_df['Label'] = test_df['Image_Label'].str.split("_", n = 1, expand = True)[1]

test_df['Image'] = test_df['Image_Label'].str.split("_", n = 1, expand = True)[0]



types = ['Fish', 'Flower', 'Gravel', 'Sugar']





#optimized

pixel_thresholds =    [0.5,   0.5,   0.5,   0.5 ]

mask_sum_threshold  = [10000, 10000, 10000, 9000]

mask_threshold=[1000, 1000, 1000, 1000]



def mask_reduce(mask):

    

    reduced_mask = np.zeros(mask.shape,np.float32)

    

    for idx in range(mask.shape[2]):

        label_num, labeled_mask = cv2.connectedComponents(mask[:,:, idx].astype(np.uint8))

        



        for label in range(1, label_num):

            single_label_mask = (labeled_mask == label)



            if single_label_mask.sum() > mask_threshold[idx]:

                reduced_mask[single_label_mask, idx] = 1



    return reduced_mask.astype('uint8')



def mask_filter(mask):

    

    lim = np.sum(mask, axis=(0,1)) < mask_sum_threshold

    

    for i in range(len(lim)):

        if lim[i]: mask[..., i] = 0

    

    return mask



def cleanup(pred):



    return (pred>pixel_thresholds).astype('uint8')



test_df.head()
path = '../input/single-models/'

models = []



model1 = sm.FPN('resnet34', encoder_weights=None, classes=4, input_shape=(None, None, 3), activation='sigmoid')

model1.load_weights(path+'Deotte-NormalBCEJaccard-FPN-Resnet34-val_loss-256.h5') # 0.6457

models.append({"model": model1, 'weight': 1})



model2 = sm.Unet('efficientnetb0', encoder_weights=None, classes=4, input_shape=(None, None, 3), activation='sigmoid')

model2.load_weights(path+'Deotte-NormalJackardBCE-NoPseudo-K0-256-ThisisGood-0.6483.h5') #0.6483

models.append({"model": model2, 'weight': 1})



model3 = sm.FPN('efficientnetb0', encoder_weights=None, classes=4, input_shape=(None, None, 3), activation='sigmoid')

model3.load_weights(path+'Deotte-NormalBCEJaccard-FPN-val_loss-256.h5') # 0.6388

models.append({"model": model3, 'weight': 1})

                                     

model4 = sm.Unet('efficientnetb0', encoder_weights=None, classes=4, input_shape=(None, None, 3), activation='sigmoid')

model4.load_weights(path+'Deotte-perImageBCEJackard_real-noPseudo--256-0.6410.h5') #0.6410

models.append({"model": model4, 'weight': 1})



#------ Pseudo

model5 = sm.FPN('efficientnetb0', encoder_weights=None, classes=4, input_shape=(None, None, 3), activation='sigmoid')

model5.load_weights(path+'PerImageBCEJaccard-FPN-pseudo-256.h5') # 0.6421

models.append({"model": model5, 'weight': 1})



model6 = sm.Unet('efficientnetb0', encoder_weights=None, classes=4, input_shape=(None, None, 3), activation='sigmoid')

model6.load_weights(path+'NormalBCEJackard-Unet-Effnet-pseudo-256.h5') #0.6481

models.append({"model": model6, 'weight': 1})



model7 = sm.FPN('resnet34', encoder_weights=None, classes=4, input_shape=(None, None, 3), activation='sigmoid')

model7.load_weights(path+'NormalBCEJaccard-FPN-Resnet34-pseudo-256.h5') # 0.6457

models.append({"model": model7, 'weight': 1})



model8 = sm.Unet('efficientnetb0', encoder_weights=None, classes=4, input_shape=(None, None, 3), activation='sigmoid')

model8.load_weights(path+'NormalBCEJackard-Unet-Effnet-pseudo-K1-256.h5') #0.6481

models.append({"model": model8, 'weight': 1})



model9 = sm.Unet('resnet34', encoder_weights=None, classes=4, input_shape=(None, None, 3), activation='sigmoid')

model9.load_weights(path+'NormalBCEJackard-Unet-Resnet34-pseudo-K1-256.h5') #0.6481

models.append({"model": model9, 'weight': 1})
!ls -l ../input/single-models/
import efficientnet.keras as efn 

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.models import Sequential



def getClassifier(name):

    base = efn.EfficientNetB3(weights=None, include_top=False, input_shape=(None, None, 3), pooling='avg')

    base.trainable=True



    dropout_dense_layer = 0.3 # for B0



    classifier_model = Sequential()

    classifier_model.add(base)

    classifier_model.add(Dropout(dropout_dense_layer))

    classifier_model.add(Dense(4, activation='sigmoid'))



    classifier_model.compile(

        loss='binary_crossentropy',

        optimizer=Adam(),

        metrics=['accuracy']

    )

    classifier_model.summary()

    classifier_model.load_weights(path+name)

    

    return classifier_model



classifier_models = []



#---- without Pseudo Labels

classifier_models.append(getClassifier('classifierB3-256.h5'))

classifier_models.append(getClassifier('classifierB3-blackout00-smooth0-256.h5'))

classifier_models.append(getClassifier('classifierB3-blackout04-256.h5'))



#---- WITH Pseudo Labels
ids = test_df['Image'].unique()

test_df.EncodedPixels = ''



height = 256

width = int(height * 1.5)



class_thresholds = [0.5, 0.5, 0.5, 0.5]



for picIdx in range(len(ids)):

    

    filename = ids[picIdx]

    img = load_image(filename)

    

    if picIdx % 100 == 0: print(picIdx)



    batch = np.zeros((4, height, width, 3))

    batch[0] = img

    batch[1] = img[ :, ::-1, :]

    batch[2] = img[ ::-1, :, :]

    batch[3] = img[ ::-1, ::-1, :]

    batch = normalize(batch)

    

    predTTA = np.zeros((batch.shape[0], img.shape[0], img.shape[1], 4))

    for j in range(len(models)):



        predTTA += models[j]['model'].predict(batch) 

        

    predTTA /= len(models)

    

    # average TTA flips

    pred = (predTTA[0, :, :, :]+predTTA[1, :, ::-1, :]+predTTA[2, ::-1, :, :]+predTTA[3, ::-1, ::-1, :])/4.0



    if len(classifier_models)>0:

        

        classpred = np.zeros((batch.shape[0], 4))

        for j in range(len(classifier_models)):

            classpred += classifier_models[j].predict(batch)

        

        classpred /= len(classifier_models)

        classpred = np.mean(classpred, axis=0)

        

        # avoid 0 masks

        if np.sum(classpred>class_thresholds) == 0:

            classpred[np.argmax(classpred)]=1

        

        #remove masks by classifier

        pred = pred * (classpred>class_thresholds)



  

    #pred_orig = pred.copy()

    pred = cleanup(pred) # argmax

    pred = mask_reduce(pred) #remove small patches

    pred = mask_filter(pred) #remove masks containing too few pixels

    pred = resizeMask(pred, 525, 350)    

    

    for myType in types:

        name = filename+"_"+myType

        line = test_df[test_df.Image_Label == name].index[0] 

        

        i=types.index(myType)

        maskrle = mask2rle(pred[..., i])        

        

        test_df.loc[line, 'EncodedPixels'] = maskrle



#Submission

sub = test_df[['Image_Label', 'EncodedPixels']]

sub.to_csv('submission.csv', index=False)

sub.head(30)
sub['Label'] = sub['Image_Label'].str.split("_", n = 1, expand = True)[1]

sub['Image'] = sub['Image_Label'].str.split("_", n = 1, expand = True)[0]



print(sub[(sub.Label == 'Fish')&(sub.EncodedPixels != '')]['Image'].count())

print(sub[(sub.Label == 'Sugar')&(sub.EncodedPixels != '')]['Image'].count())

print(sub[(sub.Label == 'Gravel')&(sub.EncodedPixels != '')]['Image'].count())

print(sub[(sub.Label == 'Flower')&(sub.EncodedPixels != '')]['Image'].count())