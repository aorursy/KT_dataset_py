%%time

import os

import matplotlib.pyplot as plt

from tqdm import tqdm

import numpy as np

import gc

import tensorflow as tf

import time

t0 = time.time()



## this script transports l5kit and dependencies

os.system('pip uninstall typing -y')

os.system('pip install --target=/kaggle/working pymap3d==2.1.0')

os.system('pip install --target=/kaggle/working protobuf==3.12.2')

os.system('pip install --target=/kaggle/working transforms3d')

os.system('pip install --target=/kaggle/working zarr')

os.system('pip install --target=/kaggle/working ptable')



os.system('pip install --no-dependencies --target=/kaggle/working l5kit')

#!pip install --upgrade pip

#!pip install pymap3d==2.1.0

#!pip install -U l5kit
import os

DIR_INPUT = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"

import os

os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/single_mode_sample_submission.csv"

MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/multi_mode_sample_submission.csv"

from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import AgentDataset, EgoDataset

from l5kit.evaluation import write_pred_csv

from l5kit.rasterization import build_rasterizer
DEBUG = False  # True just trains for 10 steps instead of the full dataset

cfg = {

    'format_version': 4,

    'model_params': {

        'model_architecture': 'resnet50',

        'history_num_frames': 20,

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

        'max_num_steps': 10*16 if DEBUG else 3200,

        'checkpoint_every_n_steps': 5000,

        'train_batch' : 32,

        'num_batch' : 10

        

        # 'eval_every_n_steps': -1

    },

    

    'test_data_loader': {

        'key': 'scenes/test.zarr',

        'batch_size': 8,

        'shuffle': False,

        'num_workers': 4

    },

    

    

    

    'valid_data_loader': {

        'key': 'scenes/validate.zarr',

        'batch_size': 8,

        'shuffle': False,

        'num_workers': 4

    },

    

    

}


train_cfg = cfg["train_data_loader"]



# Rasterizer

dm = LocalDataManager(None)

rasterizer = build_rasterizer(cfg, dm)



# Train dataset/dataloader



train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()

train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

hist_shape = train_dataset[0]['history_positions'].shape

num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2

num_in_channels = 3 + num_history_channels

num_targets = 2 * cfg["model_params"]["future_num_frames"]



print(train_dataset)





gc.collect()
dataset_path1 = dm.require(cfg["valid_data_loader"]["key"])

valid_zarr = ChunkedDataset(dataset_path1).open()

valid_dataset = AgentDataset(cfg, valid_zarr, rasterizer)





valid_itr = iter(valid_dataset)

n_valid = 100



val_inputs = np.zeros(shape=(n_valid,224,224, num_in_channels) )

val_targets = np.zeros(shape=(n_valid,num_targets))

for itr in tqdm(range(n_valid)):

    data = next(valid_itr)



    val_inputs[itr] = data['image'].transpose(1,2,0)    

    val_targets[itr] = data['target_positions'].reshape(-1,num_targets)

    gc.collect()

del valid_dataset

    
idx = 100

plt.scatter(train_dataset[idx]['history_positions'][:,0],train_dataset[idx]['history_positions'][:,1])

plt.scatter(train_dataset[idx]['target_positions'][:,0],train_dataset[idx]['target_positions'][:,1],c='r')

plt.show()

print(train_dataset[0]['target_positions'].shape) 
from keras.applications.resnet50 import ResNet50

from keras.utils.conv_utils import convert_kernel

from keras.layers import (Input, Conv2D, Flatten,Dense,AveragePooling2D,Dropout,MaxPooling2D,BatchNormalization)

from keras.models import Model, Sequential

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras import optimizers



base_in = Input(shape=(224,224,num_in_channels))

base_model=Conv2D(20,kernel_size=1,use_bias=False,padding="same")(base_in)

base_model=Conv2D(3,kernel_size=3,use_bias=False,padding="same")(base_model)





base_model = ResNet50(include_top=False,

                      weights= 'imagenet',

                      input_tensor= Input(shape = (224,224,3)),

                      pooling='max'

                ) (base_model)



#dense_model = base_model.output

dense_model = Dense(1000, activation="linear")(base_model)

dense_model = Dropout(.25)(dense_model)

dense_model = Dense(500, activation="linear")(dense_model)

dense_model = Dropout(.25)(dense_model)

dense_model = Dense(num_targets, activation="linear")(dense_model)



model = Model(inputs=base_in, outputs=dense_model)

opt = optimizers.Adam(lr=0.002)

model.compile(optimizer=opt, loss='mse')



model.summary()
import gc

MC = ModelCheckpoint('./model.h5', verbose=True,monitor='val_loss',mode='min',

        save_weights_only=True,save_best_only=True)



stop = EarlyStopping(monitor = 'val_loss', restore_best_weights=True , patience = 5)



tr_it = iter(train_dataset)

batch_size = cfg['train_params']['train_batch']

#progress_bar = tqdm(range(0,cfg["train_params"]["max_num_steps"],batch_size))

progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))

two_hours = 60 * 60 * 4

losses = []

hist = []

for itr in progress_bar:#range(0,cfg["train_params"]["max_num_steps"],batch_size):

    inputs = np.zeros(shape=(batch_size,224,224,num_in_channels))

    targets = np.zeros(shape=(batch_size, num_targets))

    

    for i in range(batch_size):

        

        try:

            data = next(tr_it)

        except StopIteration:

            tr_it = iter(train_dataset)

            data = next(tr_it)

            

        inputs[i] = data['image'].transpose(1,2,0)

        targets[i] = data['target_positions'].reshape(-1,num_targets)

   

    h = model.fit(inputs, targets,

                  batch_size = batch_size / 2 ,

                  validation_data = (val_inputs, val_targets),

                  verbose = 0,

                 callbacks = [MC, stop])

                  

    hist.append(h.history)

    gc.collect()

    # For training + submission, break if training exceeds 6 hours

    if (time.time()-t0) > four_hours:

        break

    

    

vl = [hi['val_loss'] for hi in hist]

l = [hi['loss'] for hi in hist]

plt.plot(np.log(vl), label = 'val_loss')

plt.plot(np.log(l), label = 'loss')

plt.legend(loc=0)

plt.show()
model.save('modelv0.h5')

#Example Prediction:

import matplotlib.pyplot as plt

a1 = next(tr_it)

inp = a1['image'].transpose(1,2,0)

act = a1['target_positions']

pred = model.predict(inp.reshape(-1,224,224,num_in_channels)).reshape(50,2)

plt.scatter(act[:,0], act[:,1])

plt.scatter(pred[:,0],pred[:,1])
test_cfg = cfg["test_data_loader"]



# Rasterizer

rasterizer = build_rasterizer(cfg, dm)



# Test dataset/dataloader

test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()

test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]

test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
t_shape = test_dataset[0]["target_positions"].shape

timestamps = []

agent_ids = []

coords = []

for it in tqdm(test_dataset):

    

    dat = it['image'].transpose(1,2,0)

    coords.append(np.array(model.predict(dat.reshape(1,224,224,num_in_channels)).reshape(t_shape)))

    timestamps.append(it["timestamp"])

    agent_ids.append(it["track_id"])

    

    
from l5kit.evaluation import write_pred_csv





write_pred_csv('submission.csv',

                timestamps = np.array(timestamps),

                track_ids = np.array(agent_ids),

                coords = np.array(coords) )