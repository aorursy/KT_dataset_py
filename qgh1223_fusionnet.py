import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from skimage.io import imread,imshow
from skimage.transform import resize
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
from tqdm import tqdm_notebook
from keras.models import Model,load_model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,TensorBoard

from keras import backend as K
from keras.optimizers import Adam
from keras.regularizers import l2
import keras
import tensorflow as tf
IMG_ROW=IMG_COL=128
IMG_CHANNEL=3
TRAIN_IMG_DIR='../input/train/images/'
TRAIN_MASK_DIR='../input/train/masks/'
TEST_IMG_DIR='../input/test/images/'
DEPTH_DIR='../input/depths.csv'
depth=pd.read_csv(DEPTH_DIR)
depthinfo=pd.Series(depth['z'])
depthinfo.index=depth['id']

def get_img_mask_array(imgpath,maskpath):
    #print(imgpath)
    img=imread(imgpath)[:,:,:IMG_CHANNEL]
    img=resize(img,(IMG_ROW,IMG_COL),mode='constant',
               preserve_range=True)
    mask=np.zeros((IMG_ROW,IMG_COL,1),dtype=np.bool)
    mask_=imread(maskpath)
    mask_=np.expand_dims(resize(mask_,(IMG_ROW,IMG_COL),
                                mode='constant',preserve_range=True),
                         axis=-1)
    mask=np.maximum(mask,mask_)
    return np.asarray(img),np.asarray(mask)
imgarray=np.zeros((4500,IMG_ROW,IMG_COL,IMG_CHANNEL),dtype=np.uint8)
maskarray=np.zeros((4500,IMG_ROW,IMG_COL,1),dtype=np.bool)
for k,path in enumerate(os.listdir(TRAIN_IMG_DIR)):
    if(os.path.isfile(TRAIN_IMG_DIR+path)):
        imgpath=TRAIN_IMG_DIR+path
        maskpath=TRAIN_MASK_DIR+path
        img,mask=get_img_mask_array(imgpath,maskpath)
        imgarray[k]=img
        maskarray[k]=mask
for i in range(10):
    rnd_id=random.randint(0,len(imgarray)-1)
    f,ax=plt.subplots(1,2,figsize=(15,2))
    axes=ax.flatten()
    j=0
    for ax in axes:
        if(j==0):
            ax.imshow(imgarray[rnd_id]/255,cmap='seismic')
        else:
            ax.imshow(np.squeeze(maskarray[rnd_id]),cmap='seismic')
        j+=1
img_train,img_test,mask_train,mask_test=train_test_split(np.asarray(imgarray),np.asarray(maskarray),
                                                        test_size=0.1)
print(img_train.shape)
print(mask_train.shape)
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
def lovasz_grad(gt_sorted):

    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def lovasz_hinge(logits, labels, per_image=True, ignore=None):

    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):

    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss
def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0      
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)

def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred>0.5], tf.float64)

def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred >0], tf.float64)
def double_conv_layer(x,size,dropout=0.0,batch_norm=True):
    conv=Conv2D(size,(3,3),padding='same')(x)
    if(batch_norm==True):
        conv=BatchNormalization(axis=3)(conv)
    conv=Activation('relu')(conv)
    conv=Conv2D(size,(3,3),padding='same')(conv)
    if(batch_norm==True):
        conv=BatchNormalization(axis=3)(conv)
    conv=Activation('relu')(conv)
    if(dropout!=0.0):
        conv=SpatialDropout2D(dropout)(conv)
    return conv

def build_zf_unet(filters):
    inputs=Input((IMG_ROW,IMG_COL,IMG_CHANNEL))
    conv1=double_conv_layer(inputs,filters)
    p1=MaxPooling2D(pool_size=(2,2))(conv1)
    
    conv2=double_conv_layer(p1,2*filters)
    p2=MaxPooling2D(pool_size=(2,2))(conv2)
    
    conv3=double_conv_layer(p2,4*filters)
    p3=MaxPooling2D(pool_size=(2,2))(conv3)
    
    conv4=double_conv_layer(p3,8*filters)
    p4=MaxPooling2D(pool_size=(2,2))(conv4)
    
    conv5=double_conv_layer(p4,16*filters)
    p5=MaxPooling2D(pool_size=(2,2))(conv5)
    
    conv6=double_conv_layer(p5,32*filters)
    
    up7=concatenate([UpSampling2D(size=(2,2))(conv6),conv5],axis=3)
    conv7=double_conv_layer(up7,16*filters)
    
    up8=concatenate([UpSampling2D(size=(2,2))(conv7),conv4],axis=3)
    conv8=double_conv_layer(up8,8*filters)
    
    up9=concatenate([UpSampling2D(size=(2,2))(conv8),conv3],axis=3)
    conv9=double_conv_layer(up9,4*filters)
    
    up10=concatenate([UpSampling2D(size=(2,2))(conv9),conv2],axis=3)
    conv10=double_conv_layer(up10,2*filters)
    
    up11=concatenate([UpSampling2D(size=(2,2))(conv10),conv1],axis=3)
    conv11=double_conv_layer(up11,filters,0)
    
    convfinal=Conv2D(1,(1,1))(conv11)
    convfinal=Activation('sigmoid')(convfinal)
    
    model=Model(inputs,convfinal)
    model.summary()
    return model
model=build_zf_unet(8)
model.compile(loss=lovasz_loss, optimizer='Adam', metrics=[my_iou_metric_2])

callbacks=[
    ReduceLROnPlateau(monitor='val_loss',patience=3,min_lr=1e-9,verbose=1,mode='min'),
    ModelCheckpoint('salt_model.h5',monitor='val_loss',save_best_only=True,verbose=1)
]
history=model.fit(img_train,mask_train,epochs=100,
                 validation_data=(img_test,mask_test),
                 callbacks=callbacks)
plt.plot(history.history['mean_iou'])
plt.plot(history.history['val_mean_iou'])
plt.legend(['train','valid'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','valid'])
predictarray=model.predict(imgarray)

testimgpathlist=os.listdir(TEST_IMG_DIR)
testimgarr=np.zeros((len(testimgpathlist),IMG_ROW,IMG_COL,IMG_CHANNEL))
for i,path in enumerate(os.listdir(TEST_IMG_DIR)):
    imgpath=TEST_IMG_DIR+path
    img=imread(imgpath)[:,:,:IMG_CHANNEL]
    img=resize(img,(IMG_ROW,IMG_COL),mode='constant',
               preserve_range=True)
    testimgarr[i]=img
print(testimgarr.shape)
trainmasklist=model.predict(imgarray)
testmasklist=model.predict(testimgarr)
for i in range(10):
    rnd_id=random.randint(0,len(imgarray)-1)
    f,ax=plt.subplots(1,3,figsize=(15,2))
    axes=ax.flatten()
    j=0
    for ax in axes:
        if(j==0):
            ax.imshow(imgarray[rnd_id]/255,cmap='seismic')
        elif(j==1):
            ax.imshow(np.squeeze(maskarray[rnd_id]),cmap='seismic')
        else:
            ax.imshow(np.squeeze(trainmasklist[rnd_id]),cmap='seismic')
        j+=1
for i in range(10):
    rnd_id=random.randint(0,len(testimgarr)-1)
    f,ax=plt.subplots(1,2,figsize=(15,2))
    axes=ax.flatten()
    j=0
    for ax in axes:
        if(j==0):
            ax.imshow(testimgarr[rnd_id]/255,cmap='seismic')
        else:
            ax.imshow(np.squeeze(testmasklist[rnd_id]),cmap='seismic')
        j+=1
def iou_metrics(y_true,y_pred,print_table=False):
    labels=y_true
    true_objects=2
    pred_objects=2
    temp1=np.histogram2d(labels.flatten(),y_pred.flatten())
    intersection=temp1[0]
    area_true=np.histogram2d(labels,bins=[0,0.5,1])[0]
    area_pred=np.histogram2d(y_pred,bins=[0,0.5,1])[0]
    area_true=np.expand_dims(area_true,-1)
    area_pred=np.expand_dims(area_pred,0)
    union=area_true+area_pred-intersection
    intersection=intsersection[1:,1:]
    intersection[intersection==0]=1e-9
    union=union[1:,1:]
    union[union==0]=1e-9
    iou=intersection/union
    def precision_at(thresold,iou):
        match=iou>thresold
        true_positives=np.sum(matches,axis=1)==1
        false_positives=np.sum(matches,axis=0)==0
        false_negatives=np.sum(matches,axis=1)==0
        tp,fp,fn=np.sum(true_positives),np.sum(false_positives),np.sum(false_negatives)
        return tp,fp,fn
    prec=[]
    for t in np.arange(0.5,1.0,0.05):
        tp,fp,fn=precision_at(t,iou)
        if(tp+fp+fn>0):
            p=tp/(tp+fp+fn)
        else:
            p=0
        prec.append(p)
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)
thresholds_ori = np.linspace(0.3, 0.7, 31)
thresholds = np.log(thresholds_ori/(1-thresholds_ori))
ious = np.array([iou_metric_batch(maskarray, trainmasklist > threshold) for threshold in tqdm_notebook(thresholds)])
print(ious)
threshold_best_index = np.argmax(ious) 
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()
def rle_encode(im):
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)