import tensorflow as tf
import os
import tensorflow.keras.backend as K

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,accuracy_score

import albumentations as albu

import cv2
import gc
import re
from tqdm import tqdm
from typing import Dict

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv
gpus = tf.config.list_physical_devices('GPU'); print(gpus)
if len(gpus)==1: strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
else: strategy = tf.distribute.MirroredStrategy()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
print('Mixed precision enabled')
DIR_INPUT = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"

SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/single_mode_sample_submission.csv"
MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/multi_mode_sample_submission.csv"

DEBUG = False
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager()
cfg = {
    'format_version': 4,
    'model_params': {
        'model_architecture': 'resnet50',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },
    
    'raster_params': {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },
    
    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 12,
        'shuffle': True,
        'num_workers': 4
    },
    
    'train_params': {
        'max_num_steps':10000,
        'checkpoint_every_n_steps': 5000,
        
        # 'eval_every_n_steps': -1
    },
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 4
    }
}

train_cfg = cfg["train_data_loader"]

# Rasterizer
rasterizer = build_rasterizer(cfg, dm)

# Train dataset/dataloader
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
# ===== INIT DATASET
test_cfg = cfg["test_data_loader"]

# Rasterizer
rasterizer = build_rasterizer(cfg, dm)

# Test dataset/dataloader
test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()
test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]
test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
# c=0
# for i in tqdm(test_dataset):
#     c+=1
# print(c)
class DataGen(tf.keras.utils.Sequence):
    'helps to generate data, keras'
    def __init__(self,_size,batch_size=128,agent_ids=None,timestamps=None,train=True,train_dataset=None,test_dataset=None):
        
        self.batch_size=batch_size
        self._size=_size
        self.train=train
        
        if train:
            self.train_data_iter=iter(train_dataset)
            self.size = self._size // self.batch_size
            self.size += int(self._size % self.batch_size!=0)
            
        else:
            self.test_data_iter=iter(test_dataset)
            self.size = self._size // self.batch_size
            self.size += int(self._size % self.batch_size!=0)
        #self.indexes = np.arange(train_size)
        
        self.on_epoch_end()
        
        
    def __len__(self):
        return self.size
    
    
    def __getitem__(self,index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        if index==self.size-1:
            indexes=self.indexes[index*self.batch_size:(index*self.batch_size)+((self._size)%self.batch_size)]
        images=np.zeros((len(indexes),224,224,25))
        targets=np.zeros((len(indexes),100))
        if self.train:
            for index in range(len(indexes)):
#                 data=next(self.train_data_iter)
                try:
                    data = next(self.train_data_iter)
                except StopIteration:
                    self.train_data_iter=iter(train_dataset)
                    data = next(self.train_data_iter)
                images[index,:,:,:]=data['image'].transpose(1,2,0)#[:,:,:25]
                targets[index,:]=data["target_positions"].reshape((100,))
                #targets[index,:,:]=np.expand_dims(data["target_availabilities"], axis=-1)
            return images,targets
        else:
            for index in range(len(indexes)):
                try:
                    data = next(self.test_data_iter)
                except StopIteration:
                    self.test_data_iter=iter(test_dataset)
                    data = next(self.test_data_iter)
                images[index,:,:,:]=data['image'].transpose(1,2,0)#[:,:,:25]
                timestamps.append(data["timestamp"])
                agent_ids.append(data['track_id'])
            return images
            
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self._size)
gen=DataGen(100,batch_size=8,train=True,train_dataset=train_dataset)
c=iter(gen)
next(c)[0].shape
next(c)[1].shape
#!pip install efficientnet 
#import efficientnet.tfkeras as efn
#from tensorflow.keras.applications.vgg19 import VGG19
# VERSION MAJOR and MINOR for logging
mm = 0; rr = 0

# BEGIN LOG FILE
f = open(f'log-{mm}-{rr}.txt','a')
print('Logging to "log-%i-%i.txt"'%(mm,rr))
f.write('#############################\n')
f.write(f'Trial mm={mm}, rr={rr}\n')
f.write('ResNet50, batch_size=256, seed=42, 224x224, lr=5e-5\n')
f.write('#############################\n')
f.close()

batch_size = 256
DIM = 224
num_targets = 2 * cfg["model_params"]["future_num_frames"]
train_data_size=20000
test_data_size=71122
def build_model():
    
    inp = tf.keras.Input(shape=(DIM,DIM,25))
    inp2=tf.keras.layers.Conv2D(16,kernel_size=1,use_bias=False,padding="same")(inp)
    inp3=tf.keras.layers.Conv2D(8,kernel_size=3,use_bias=False,padding="same")(inp2)
    inp4=tf.keras.layers.Conv2D(5,kernel_size=3,use_bias=False,padding="same")(inp3)
    inp5=tf.keras.layers.Conv2D(3,kernel_size=3,use_bias=False,padding="same")(inp4)
#     base_model = efn.EfficientNetB4(weights=None,include_top=False, input_shape=(DIM,DIM,3)) 
    base_model=tf.keras.applications.ResNet50(include_top=False,weights="imagenet",input_shape=(224,224,3))
    x = base_model(inp5)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)    
    out = tf.keras.layers.Dense(num_targets, activation='softmax',name='out',dtype='float32')(x)
    
    model = tf.keras.Model(inputs=inp, outputs=out)
    opt = tf.keras.optimizers.Adam(lr=1e-3)
    model.compile(loss='mean_squared_error', optimizer = opt,\
              metrics=['mean_absolute_error'])
        
    return model
K.clear_session()
with strategy.scope():
    model=build_model()
model.summary()
def get_lr_callback(batch_size=64):
    lr_start   = 0.00001
    lr_max     = 0.0000078125 * batch_size
    lr_min     = 0.000001
    lr_ramp_ep = 3
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr
#     if plot:
#         y=[lrfnf(i) for i in range(20)]
#         x=[i for i in range(20)]
#         plt.plot(x,y)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
    return lr_callback
sv = tf.keras.callbacks.ModelCheckpoint(
        './model.h5', verbose=True,monitor='loss',mode='min',
        save_weights_only=True,save_best_only=True)
train_gen=DataGen(train_data_size,batch_size=128,train=True,train_dataset=train_dataset)
model.fit(train_gen,epochs=1,callbacks=[sv],verbose=1,batch_size=128)#,get_lr_callback(batch_size=128)
del train_gen
del train_dataset
gc.collect()
timestamps = []
agent_ids = []
test_gen=DataGen(test_data_size,batch_size=batch_size,agent_ids=agent_ids,timestamps=timestamps,train=False,test_dataset=test_dataset)
pred=model.predict(test_gen,verbose=1,batch_size=batch_size)
print(pred.shape)
def reshape_test(arr):
    return arr.reshape((50,2))
%time
preds=np.apply_along_axis(reshape_test, 1, pred)
preds.shape
write_pred_csv('submission.csv',
               timestamps=np.array(timestamps[:test_data_size]),
               track_ids=np.array(agent_ids[:test_data_size]),
               coords=preds)
!mkdir ./sv_model_sub
print('Loading best model...')
model.load_weights('model.h5')
timestamps = []
agent_ids = []
pred=model.predict(test_gen,verbose=1,batch_size=batch_size)
del test_dataset
del test_gen
del model
gc.collect()
%time
preds=np.apply_along_axis(reshape_test, 1, pred)
preds.shape
write_pred_csv('submission.csv',
               timestamps=np.array(timestamps[:test_data_size]),
               track_ids=np.array(agent_ids[:test_data_size]),
               coords=preds)
