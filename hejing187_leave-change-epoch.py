!pip install -q efficientnet
import os 
import gc
import re

import cv2
import math
import numpy as np
import scipy as sp
import pandas as pd

import tensorflow as tf
from IPython.display import SVG
import efficientnet.tfkeras as efn
from keras.utils import plot_model
import tensorflow.keras.layers as L
from keras.utils import model_to_dot
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from kaggle_datasets import KaggleDatasets
from tensorflow.keras.applications import DenseNet121

import seaborn as sns
from tqdm import tqdm
import matplotlib.cm as cm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

tqdm.pandas()
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(0)
tf.random.set_seed(0)

import warnings
warnings.filterwarnings('ignore')
EPOCHS=40
SAMPLE_LEN=100
IMAGE_PATH='../input/plant-pathology-2020-fgvc7/images/'
TEST_PATH='../input/plant-pathology-2020-fgvc7/test.csv'
TRAIN_PATH='../input/plant-pathology-2020-fgvc7/train.csv'
SUB_PATH='../input/plant-pathology-2020-fgvc7/sample_submission.csv'

sub=pd.read_csv(SUB_PATH)
train_data=pd.read_csv(TRAIN_PATH)
test_data=pd.read_csv(TEST_PATH)
AUTO=tf.data.experimental.AUTOTUNE
tpu=tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy=tf.distribute.experimental.TPUStrategy(tpu)

BATCH_SIZE=16*strategy.num_replicas_in_sync
GCS_DS_PATH=KaggleDatasets().get_gcs_path()
def format_path(st):
    return GCS_DS_PATH+'/images/'+st+'.jpg'

test_paths=test_data.image_id.apply(format_path).values
train_paths=train_data.image_id.apply(format_path).values

train_labels=np.float32(train_data.loc[:,'healthy':'scab'].values)
train_paths,valid_paths,train_labels,valid_labels=\
train_test_split(train_paths,train_labels,test_size=0.15,random_state=2020)
def decode_image(filename,label=None,image_size=(800,800)):
    bits=tf.io.read_file(filename)
    image=tf.image.decode_jpeg(bits,channels=3)
    image=tf.cast(image,tf.float32)/255.0
    image=tf.image.resize(image,image_size)
    
    if label is None:
        return image
    else:
        return image,label
    
def data_augment(image,label=None):
    image=tf.image.random_flip_left_right(image)
    image=tf.image.random_flip_up_down(image)
    image=tf.image.rot90(image)
    image=tf.image.transpose(image)
    image=tf.image.random_brightness(image,0.2)
    image=tf.image.random_contrast(image,0.2,0.5)
    image=tf.image.random_hue(image,0.2)
    
    if label is None:
        return image
    else:
        return image,label
                                  

     #help(tf.image)
#gridmask

IMG_DIM=(800,800)
AugParams={
     'd1':100,
     'd2':160,
     'rotate':45,
     'ratio':0.5
}

def transform(image,inv_mat,image_shape):
    h,w,c=image_shape
    cx,cy=w//2,h//2
    
    new_xs=tf.repeat(tf.range(-cx,cx,1),h)
    new_ys=tf.tile(tf.range(-cy,cy,1),[w])
    new_zs=tf.ones([h*w],dtype=tf.int32)
    
    old_coords=tf.matmul(inv_mat,tf.cast(tf.stack([new_xs,new_ys,new_zs]),tf.float32))
    old_coords_x,old_coords_y=tf.round(old_coords[0,:]+w//2),tf.round(old_coords[1,:]+h//2)
    
    clip_mask_x=tf.logical_or(old_coords_x<0,old_coords_x>w-1)
    clip_mask_y=tf.logical_or(old_coords_y<0,old_coords_y>h-1)
    clip_mask=tf.logical_or(clip_mask_x,clip_mask_y)
    
    old_coords_x=tf.boolean_mask(old_coords_x,tf.logical_not(clip_mask))
    old_coords_y=tf.boolean_mask(old_coords_y,tf.logical_not(clip_mask))
    new_coords_x=tf.boolean_mask(new_xs+cx,tf.logical_not(clip_mask))
    new_coords_y=tf.boolean_mask(new_ys+cy,tf.logical_not(clip_mask))
    
    old_coords=tf.cast(tf.stack([old_coords_y,old_coords_x]),tf.int32)
    new_coords=tf.cast(tf.stack([new_coords_y,new_coords_x]),tf.int64)
    rotated_image_values=tf.gather_nd(image,tf.transpose(old_coords))
    rotated_image_channel=list()
    for i in range(c):
        vals=rotated_image_values[:,i]
        sparse_channel=tf.SparseTensor(tf.transpose(new_coords),vals,[h,w])
        rotated_image_channel.append(tf.sparse.to_dense(sparse_channel,default_value=0,validate_indices=False))
    return tf.transpose(tf.stack(rotated_image_channel),[1,2,0])

def random_rotate(image,angle,image_shape):
    def get_rotation_mat_inv(angle):
        angle=math.pi*angle/180
        
        cos_val=tf.math.cos(angle)
        sin_val=tf.math.sin(angle)
        one=tf.constant([1],tf.float32)
        zero=tf.constant([0],tf.float32)
        
        rot_mat_inv=tf.concat([cos_val,sin_val,zero,
                                 -sin_val,cos_val,zero,
                                 zero,zero,one],axis=0)
        rot_mat_inv=tf.reshape(rot_mat_inv,[3,3])
        
        return rot_mat_inv
    
    angle=float(angle)*tf.random.normal([1],dtype='float32')
    rot_mat_inv=get_rotation_mat_inv(angle)
    return transform(image,rot_mat_inv,image_shape)

def GridMask(image_height,image_width,d1,d2,rotate_angle=1,ration=0.5):
    h,w=image_height,image_width
    hh=int(np.ceil(np.sqrt(h*h+w*w)))
    hh=hh+1 if hh%2==1 else hh
    d=tf.random.uniform(shape=[],minval=d1,maxval=d2,dtype=tf.int32)
    l=tf.cast(tf.cast(d,tf.float32)*ration+0.5,tf.int32)
    
    st_h=tf.random.uniform(shape=[],minval=0,maxval=d,dtype=tf.int32)
    st_w=tf.random.uniform(shape=[],minval=0,maxval=d,dtype=tf.int32)
    
    y_ranges=tf.range(-1*d+st_h,-1*d+st_h+1)
    x_ranges=tf.range(-1*d+st_w,-1*d+st_w+1)
    
    for i in range(0,hh//d+1):
        s1=i*d+st_h
        s2=i*d+st_w
        x_ranges=tf.concat([y_ranges,tf.range(s1,s1+1)],axis=0)
        x_ranges=tf.concat([x_ranges,tf.range(s2,s2+1)],axis=0)
        
    x_clip_mask=tf.logical_or(x_ranges<0,x_ranges>hh-1)
    y_clip_mask=tf.logical_or(y_ranges<0,y_ranges>hh-1)
    clip_mask=tf.logical_or(x_clip_mask,y_clip_mask)
    
    x_ranges=tf.boolean_mask(x_ranges,tf.logical_not(clip_mask))
    y_ranges=tf.boolean_mask(y_ranges,tf.logical_not(clip_mask))
    
    hh_ranges=tf.tile(tf.range(0,hh),[tf.cast(tf.reduce_sum(tf.ones_like(x_ranges)),tf.int32)])
    x_ranges=tf.repeat(x_ranges,hh)
    y_ranges=tf.repeat(y_ranges,hh)
    
    y_hh_indices=tf.transpose(tf.stack([y_ranges,hh_ranges]))
    x_hh_indices=tf.transpose(tf.stack([hh_ranges,x_ranges]))
    
    y_mask_sparse=tf.SparseTensor(tf.cast(y_hh_indices,tf.int64),tf.zeros_like(y_ranges),[hh,hh])
    y_mask=tf.sparse.to_dense(y_mask_sparse,1,False)
    
    x_mask_sparse=tf.SparseTensor(tf.cast(x_hh_indices,tf.int64),tf.zeros_like(x_ranges),[hh,hh])
    x_mask=tf.sparse.to_dense(x_mask_sparse,1,False)
    
    mask=tf.expand_dims(tf.clip_by_value(x_mask+y_mask,0,1),axis=-1)
    
    mask=random_rotate(mask,rotate_angle,[hh,hh,1])
    mask=tf.image.crop_to_bounding_box(mask,(hh-h)//2,(hh-w)//2,image_height,image_width)
    return mask

def apply_grid_mask(image,image_shape):
    mask=GridMask(image_shape[0],
                 image_shape[1],
                 AugParams['d1'],
                 AugParams['d2'],
                 AugParams['rotate'],
                 AugParams['ratio'])
    if image_shape[-1]==3:
        mask=tf.concat([mask,mask,mask],axis=-1)
        
    return image*tf.cast(mask,tf.float32)

def augmentation(image,label=None):
    if tf.random.uniform(shape=[],minval=0.0,maxval=1.0)>=0.5:
        image=apply_grid_mask(image,(*IMG_DIM,3))
    if label==None:
        return tf.cast(image,tf.float32)
    else:
        return tf.cast(image,tf.float32),label
    
train_dataset=(
   tf.data.Dataset
   .from_tensor_slices((train_paths,train_labels))
   .map(decode_image,num_parallel_calls=AUTO)
   .map(data_augment,num_parallel_calls=AUTO)
   .repeat()
   .shuffle(800)
   .batch(BATCH_SIZE)
   .prefetch(AUTO))

valid_dataset=(
  tf.data.Dataset
  .from_tensor_slices((valid_paths,valid_labels))
  .map(decode_image,num_parallel_calls=AUTO)
  .batch(BATCH_SIZE)
  .cache()
  .prefetch(AUTO))

test_dataset=(
   tf.data.Dataset
   .from_tensor_slices(test_paths)
   .map(decode_image,num_parallel_calls=AUTO)
   .batch(BATCH_SIZE))
#help(tf.data.Dataset)
#train_dataset
valid_nolabel=(
    tf.data.Dataset
    .from_tensor_slices(valid_paths)
    .map(decode_image,num_parallel_calls=AUTO)
    .batch(BATCH_SIZE))
def build_lrfn(lr_start=0.00001,lr_max=0.000075,
               lr_min=0.000001,lr_rampup_epochs=20,
              lr_sustain_epochs=0,lr_exp_decay=.8):
    lr_max=lr_max*strategy.num_replicas_in_sync
    
    def lrfn(epoch):
        if epoch<lr_rampup_epochs:
            lr=(lr_max-lr_start)/lr_rampup_epochs*epoch+lr_start
        elif epoch<lr_rampup_epochs+lr_sustain_epochs:
            lr=lr_max
        else:
            lr=(lr_max-lr_min)*\
               lr_exp_decay**(epoch-lr_rampup_epochs\
                             -lr_sustain_epochs)+lr_min
        return lr
    return lrfn

from keras.callbacks import ModelCheckpoint
ch_p=ModelCheckpoint(filepath='model_ef.h5',monitor='val_loss',save_weights_only=True,verbose=1)
#cosine_schedule_with_warmup
WARMUP=15
LR=0.0008
def get_cosine_schedule_with_warmup(lr,num_warmup_steps,num_training_steps,num_cycles=0.5):
    def lrfn(epoch):
        if epoch<num_warmup_steps:
            return float(epoch)/float(max(1,num_warmup_steps))*lr
        progress=float(epoch-num_warmup_steps)/float(max(1,num_training_steps-num_warmup_steps))
        return max(0.0,0.5*(1.0+math.cos(math.pi*float(num_cycles)*2.0*progress)))*lr
    return tf.keras.callbacks.LearningRateScheduler(lrfn,verbose=True)

lr_schedule=get_cosine_schedule_with_warmup(lr=LR,num_warmup_steps=WARMUP,num_training_steps=EPOCHS)
lrfn=build_lrfn()
STEPS_PER_EPOCH=train_labels.shape[0]//BATCH_SIZE
lr_schedule=tf.keras.callbacks.LearningRateScheduler(lrfn,verbose=1)
import keras.backend as K
def categorical_focal_loss(gamma=2.0,alpha=0.25):
    def focal_loss(y_true,y_pred):
        epsilon=K.epsilon()
        y_pred=K.clip(y_pred,epsilon,1.0-epsilon)
        cross_entropy=-y_true*K.log(y_pred)
        weight=alpha*y_true*K.pow((1-y_pred),gamma)
        loss=weight*cross_entropy
        loss=K.sum(loss,axis=1)
        return loss
    return focal_loss
from keras import regularizers
import tensorflow.keras as keras
def get_model():
    model=keras.Sequential()
    model.add(DenseNet121(input_shape=(800,800,3),
                         weights='imagenet',
                         include_top=False))
    model.add(L.GlobalAveragePooling2D())
    model.add(L.Dense(train_labels.shape[1],activation='softmax'))
                     #kernel_regularizer=regularizers.l2(0.01)))
    model.summary
    return model
from keras import regularizers
import tensorflow.keras as keras
def get_model():
    model=keras.Sequential()
    model.add(tf.keras.applications.DenseNet201(input_shape=(800,800,3),
                         weights='imagenet',
                         include_top=False))
    model.add(L.GlobalAveragePooling2D())
    model.add(L.Dense(train_labels.shape[1],activation='softmax',
                     kernel_regularizer=regularizers.l2(0.01)))
    model.summary
    return model
from keras import regularizers
import tensorflow.keras as keras
def get_model():
    model=keras.Sequential()
    model.add(keras.applications.InceptionResNetV2(input_shape=(800,800,3),
                               weights='imagenet',
                               include_top=False))
    model.add(L.GlobalAveragePooling2D())
    model.add(L.Dense(train_labels.shape[1],activation='softmax'))
                     #kernel_regularizer=regularizers.l2(0.01)))
    model.summary
    return model

with strategy.scope():
    model=get_model()
    model.compile(optimizer='adam',
                 loss=['categorical_crossentropy'],
                 metrics=['categorical_accuracy'])

    
    #help(tf.keras.Sequential)
    #help(regularizers)
    #help(model.compile)
    #help(tf.losses.categorical_crossentropy)
    #help(label_smoothing)
#focal_loss
with strategy.scope():
    model=get_model()
    model.compile(optimizer='adam',
                 loss=[categorical_focal_loss(gamma=2.0,alpha=0.25)],
                 metrics=['categorical_accuracy'])
history=model.fit(train_dataset,
                 epochs=EPOCHS,
                 callbacks=[lr_schedule,ch_p],
                 steps_per_epoch=STEPS_PER_EPOCH,
                 validation_data=valid_dataset)
def display_training_curves(training,validation,yaxis):
    if yaxis=='loss':
        ylabel='Loss'
        title='Loss vs. Epochs'
    else:
        ylabel='Accuracy'
        title='Accuracy vs. Epochs'
        
    fig=go.Figure()
    
    fig.add_trace(
        go.Scatter(x=np.arange(1,EPOCHS+1),mode='lines+markers',y=training,marker=dict(color='dodgerblue'),
                name='Train'))
    fig.add_trace(
        go.Scatter(x=np.arange(1,EPOCHS+1),mode='lines+markers',y=validation,marker=dict(color='darkorange'),
                name='Val'))
    fig.update_layout(title_text=title,yaxis_title=ylabel,xaxis_title='Epochs',template='plotly_white')
    fig.show()
display_training_curves(
   history.history['categorical_accuracy'],
   history.history['val_categorical_accuracy'],
   'accuracy')
display_training_curves(
   history.history['loss'],
   history.history['val_loss'],
   'loss')
def val_class(pred):
    x=len(pred)
    y=[]
    for i in range(x):
        m=np.argmax(pred[i])
        if m==0:
            y.append('healthy')
        elif m==1:
            y.append('multiple_diseases')
        elif m==2:
            y.append('rust')
        elif m==3:
            y.append('scab')
    return y
    
from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm,classes,
                         normalize=False,
                         title='Comfusion matrix',
                         cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes)
    plt.yticks(tick_marks,classes)
    
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        
    thresh=cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                horizontalalignment='center',
                color='white' if cm[i,j]>thresh else 'black')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
Y_pred=model.predict(valid_nolabel)
Y_pred=Y_pred.tolist()
Y_pred_classes=val_class(Y_pred)
valid_label=valid_labels.tolist()
Y_true_classes=val_class(valid_label)
confusion_mtx=confusion_matrix(Y_true_classes,Y_pred_classes)
plot_confusion_matrix(confusion_mtx,classes=['healthy','multiple_diseases','rust','scab'])

Y_pred_classes

Y_pred_classes_errors=Y_pred_classes[errors]
Y_pred_errors=Y_pred[errors]
Y_true_classes_errors=valid_label[errors]
X_val_errors=valid_nolabel[errors]

def display_errors(errors_index,img_errors,pred_errors,obs_errors):
    n=0
    nrows=2
    ncols=3
    fig,ax=plt.subplot(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(cols):
            error=errors_index[n]
            ax[row,col].imshow((img_errors[errors]).reshape((28,28)))
            ax[row,col].set_title('Predicted label:{}\nTrue label:{}'.format(pred_errors[error],obs_errors[error]))
            n+=1
            
Y_pred_errors_prob=np.max(Y_pred_errors,axis=1)
true_prob_errors=np.diagonal(np.take(Y_pred_errors,Y_true_errors,axis=1))
delta_pred_true_errors=Y_pred_errors_prob-true_prob_errors
sorted_delta_errors=np.argsort(delta_pred_true_errors)
most_important_errors=sorted_dela_errors[-6:]
display_errors(most_important_errors,X_val_errors,Y_pred_classes_errors,Y_true_errors)
                             
probs_dnn=model.predict(test_dataset,verbose=1)
sub.loc[:,'healthy':]=probs_dnn
sub.to_csv('/kaggle/working/submission_DenseNet121_gridmask_noL2.csv',index=False)
#TTA
TTA=4
test_pred_tta=np.zeros((len(test_data),4))
for i in range(TTA):
    test_dataset_tta=(tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image,num_parallel_calls=AUTO)
    .map(data_augment,num_parallel_calls=AUTO)
    .batch(BATCH_SIZE))
    test_pred_tta+=model.predict(test_dataset_tta,verbose=1)
submission_df=pd.read_csv(SUB_PATH)
submission_df[['healthy','multiple_diseases','rust','scab']]=test_pred_tta/TTA
submission_df.to_csv('/kaggle/working/submission_tta_IRV.csv',index=False)
submission_df.to_csv('/kaggle/working/submission_tta_cosine.csv',index=False)
#Kfold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
def prepare_train(train_paths,train_labels):
    data=(
       tf.data.Dataset
       .from_tensor_slices((train_paths,train_labels))
       .map(decode_image,num_parallel_calls=AUTO)
       .map(data_augment,num_parallel_calls=AUTO)
       .repeat()
       .shuffle(512)
       .batch(BATCH_SIZE)
       .prefetch(AUTO))
    return data

def prepare_test(test_paths):
    data=(
       tf.data.Dataset
       .from_tensor_slices((test_paths))
       .map(decode_image,num_parallel_calls=AUTO)
       .batch(BATCH_SIZE))
    return data


def prepare_val(val_paths,val_labels):
    data=(
      tf.data.Dataset
      .from_tensor_slices((val_paths,val_labels))
      .map(decode_image,num_parallel_calls=AUTO)
      .batch(BATCH_SIZE)
      .prefetch(AUTO))
    return data
    

def TPU():
    try:
        tpu=tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU',tpu.master())
    except ValueError:
        tpu=None
        
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy=tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy=tf.distribute.get_strategy()
        
    print('REPLICAS:',strategy.num_replicas_in_sync)
    return strategy

strategy=TPU()
FOLDS=5 
SEED=42
skf=StratifiedKFold(n_splits=FOLDS,shuffle=True,random_state=SEED)
test_pred=[]
val_roc_auc=[]

for i,(train_idx,val_idx) in enumerate(skf.split(train_paths,train_labels.argmax(1))):
    print();print('#'*25)
    print('### FOLD',i+1)
    print('#'*25)
    X_train,X_val=train_paths[train_idx],train_paths[val_idx]
    y_train,y_val=train_labels[train_idx],train_labels[val_idx]
    history=model.fit(
                    prepare_train(X_train,y_train),
                    steps_per_epoch=y_train.shape[0]//BATCH_SIZE,
                    validation_data=prepare_val(X_val,y_val),
                    validation_steps=y_val.shape[0]//BATCH_SIZE,
                    callbacks=[lr_schedule],
                    epochs=EPOCHS,
                    verbose=1)
    test_pred.append(model.predict(prepare_test(test_paths),verbose=1))
    val_roc_auc.append(roc_auc_score(y_val,model.predict(prepare_val(X_val,y_val),verbose=1)))
val_roc_auc
best_2_models=test_pred[3]*.7+test_pred[4]*.3
sub.iloc[:,1:]=best_2_models
sub
sub.to_csv('submission_5kfold.csv',index=False)