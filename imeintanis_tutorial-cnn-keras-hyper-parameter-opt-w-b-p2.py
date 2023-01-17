!pip install --upgrade wandb
import os 
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline
sns.set(style='white', context='notebook', palette='deep')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import gc

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import *

import tensorflow_addons as tfa

import wandb
from wandb.keras import WandbCallback
# Load the data
def load_data(path):

    train = pd.read_csv(path+"train.csv")
    test = pd.read_csv(path+"test.csv")
    
    x_tr = train.drop(labels=["label"], axis=1)
    y_tr = train["label"]
    
    print(f'Train: we have {x_tr.shape[0]} images with {x_tr.shape[1]} features and {y_tr.nunique()} classes')
    print(f'Test: we have {test.shape[0]} images with {test.shape[1]} features')
    
    return x_tr, y_tr, test


def seed_all(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
# Build CNN model 
# CNN architechture: In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

def build_model(config):
    
    fs = config.filters         # 32
    k1 = config.kernel_1        #[(5,5), (3,3)]
    k2 = config.kernel_2        # [(5,5), (3,3)]
    pad = config.padding
    activ = config.activation   # 'relu'
    pool = config.pooling       # (2,2)
    dp = config.dropout         # 0.25
    dp_out = config.dropout_f   # 0.5
    dense_units = config.dense_units  # 256
    batch_norm = False
    
    inp = Input(shape=(28,28,1))    # IMG_H, IMG_W, NO_CHANNELS
    
    # layer-1:: CNN-CNN-(BN)-Pool-dp
    x = Conv2D(filters=fs, kernel_size=k1, padding=pad, activation=activ)(inp)
    x = Conv2D(filters=fs, kernel_size=k1, padding=pad, activation=activ)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(dp)(x)    
    
    # layer-2:: CNN-CNN-(BN)-Pool-dp
    x = Conv2D(filters=fs*2, kernel_size=k2, padding=pad, activation=activ)(inp)
    x = Conv2D(filters=fs*2, kernel_size=k2, padding=pad, activation=activ)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = Dropout(dp)(x)  
    
    x = Flatten()(x)
    #     x = GlobalAveragePooling2D()(x)
    
    # FC head
    x = Dense(dense_units, activation=activ)(x)
    x = Dropout(dp_out)(x)
    
    out = Dense(10, activation="softmax")(x)
    
    model = tf.keras.models.Model(inp, out)
    
    print(model.summary())
    return model
def build_lenet(config):
    
    fs = config.filters       # 32     
    k1 = config.kernel_1      # 3  
    k2 = config.kernel_2          
    pad = config.padding
    activ = config.activation     
    dp = config.dropout           
    
    inp = Input(shape=(28,28,1))  # (IMG_H, IMG_W, NO_CHANNELS)
    
    x = Conv2D(fs, kernel_size = k1, activation=activ)(inp)
    x = BatchNormalization()(x)
    x = Conv2D(fs, kernel_size = k1, activation=activ)(x)
    x = BatchNormalization()(x)
    x = Conv2D(fs, kernel_size = 5, strides=2, padding='same', activation=activ)(x)
    x = BatchNormalization()(x)
    x = Dropout(dp)(x)
    
    x = Conv2D(fs*2, kernel_size = k1, activation=activ)(x)
    x = BatchNormalization()(x)
    x = Conv2D(fs*2, kernel_size = k1, activation=activ)(x)
    x = BatchNormalization()(x)
    x = Conv2D(fs*2, kernel_size = 5, strides=2, padding='same', activation=activ)(x)
    x = BatchNormalization()(x)
    x = Dropout(dp)(x)
    
    x = Conv2D(fs*4, kernel_size = 4, activation=activ)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(dp)(x)
    
    out = Dense(10, activation='softmax')(x)
    
    model = tf.keras.models.Model(inp, out)

    print(model.summary())
    return model
DEBUG = True          # set to True in case of setup/testing/debugging -- False when you ready to run experiments
# DATA_AUGM = False   # set to True if you wish to add data augmentation 

BATCH_SIZE = 64

if DEBUG:
    EPOCHS = 3          
else: 
    EPOCHS = 40
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("API_key")
!wandb login $api_key
# hyperparams = dict(
#      filters = 32,
#      kernel_1 = (5,5),
#      kernel_2 = (3,3),
#      padding = 'same',
#      pooling = (2,2),
#      lr = 0.001,
#      wd = 0.0,
#      lr_schedule = 'RLR',    # cos, cyclic, step decay
#      optimizer = 'Adam',     # RMS
#      dense_units=256,
#      activation='relu',      # elu, LeakyRelu
#      dropout = 0.25,
#      dropout_f = 0.5,
#      batch_size = BATCH_SIZE,
#      epochs = EPOCHS,
#  )

# wandb.init(project="kaggle-titanic", config=hyperparams)
# config = wandb.config
def train():
    
    
    hyperparams = dict(
        filters=32,
        kernel_1=(5,5),
        kernel_2=(3,3),
        padding='same',
        pooling=(2,2),
        lr=0.001,
        wd=0.0,
        lr_schedule='RLR',    # cos, cyclic, step decay
        optimizer='Adam',     # 'RMS'
        dense_units=256,
        activation='relu',      # elu, LeakyRelu
        dropout=0.25,
        dropout_f=0.5,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS)
    
    wandb.init(project="kaggle-mnist", config=hyperparams)
    config = wandb.config
    
    
    SEED = 26
    seed_all(SEED)
    
    #     # Define image sizes and reshape to a 3-dim tensor
    #     global
    IMG_H, IMG_W = 28, 28
    NO_CHANNELS = 1           # for greyscale images
    
    # load data
    x_train, y_train, x_test = load_data(path="../input/digit-recognizer/")
    
    
    # Normalize the data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Reshape to a 4-dim tensor
    x_train = x_train.values.reshape(-1, IMG_H, IMG_W, NO_CHANNELS)
    x_test = x_test.values.reshape(-1, IMG_H, IMG_W, NO_CHANNELS)
    
    # Encode labels
    y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes=10)

    print('Tensor shape (train): ', x_train.shape)
    print('Tensor shape (test): ', x_test.shape)
    print('Tensor shape (target ohe): ', y_train_ohe.shape)
    
    # Split the train and the validation set for the fitting
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train_ohe, test_size=0.15, random_state=SEED)
    
    print('Tensors shape (train):', x_tr.shape, y_tr.shape)
    print('Tensors shape (valid):', x_val.shape, y_val.shape)
    
    print('Build architecture 1')
    model = build_model(config=config)
    
#     print('Build architecture 2 - LeNet5')
#     model = build_lenet(config=config)
    
    
    # Define the optimizer
    if config.optimizer=='Adam':
        opt = Adam(config.lr)
    elif config.optimizer=='RMS':
        opt = RMSprop(lr=config.lr, rho=0.9, epsilon=1e-08, decay=0.0)
    elif config.optimizer=='Adam+SWA':
        opt = Adam(LR)
        opt = tfa.optimizers.SWA(opt)
    else: 
        opt = 'adam'    # native adam optimizer 
    
    
    # Compile the model
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    
    # Set callbacks

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, verbose=1), 
        ReduceLROnPlateau(monitor='val_accuracy', patience=10, verbose=1, factor=0.5, min_lr=1e-4),
        WandbCallback(monitor='val_loss', validation_data=(x_val, y_val))]  
    
    model.fit(x_train, y_train_ohe, 
                     batch_size=config.batch_size,    # BATCH_SIZE, 
                     epochs=config.epochs,            # EPOCHS, 
                     validation_data=(x_val, y_val), 
                     callbacks=callbacks,
                     verbose=1) 
sweep_config = {
#     'program': 'train.py',     # 'tutorial-cnn-keras-hyperparameter-opt-w-b-p2.ipynb',
    'method': 'random',         # 'grid', 'hyperopt', 'bayesian'
    'metric': {
        'name': 'val_loss',     # or 'val_accuracy'
        'goal': 'minimize'      # 'maximize'
    },
    'parameters': {
        'filters': {
            'values': [16, 32, 64]
        },
        'lr': {
            'distribution': 'uniform',
            'min': 0.0005,
            'max': 0.002
        },
        'dp': {
            'values': [0.4, 0.5]
        }
    }
}
sweep_id = wandb.sweep(sweep_config, entity="ime", project='kaggle-mnist')
wandb.agent(sweep_id, function=train, count=3, project='kaggle-mnist')   #
# !wandb agent ime/kaggle-mnist/jp5i153v --count 3 -p 'kaggle-mnist'
# !wandb agent ime/kaggle-mnist/nkf2bmi7 --count 5 -p kaggle-mnist -e ime