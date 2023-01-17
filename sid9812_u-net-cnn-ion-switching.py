import os

import matplotlib.pyplot as plt

import glob

import numpy as np

import pandas as pd

import tensorflow as tf

from keras.layers import Dense, Dropout, Reshape, Conv1D, BatchNormalization, Activation, AveragePooling1D, GlobalAveragePooling1D, Lambda, Input, Concatenate, Add, UpSampling1D, Multiply

from keras.models import Model

from keras.objectives import mean_squared_error

from keras import backend as K

from keras.losses import binary_crossentropy, categorical_crossentropy,sparse_categorical_crossentropy

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler

from keras.initializers import random_normal

from keras.optimizers import Adam, RMSprop, SGD

from keras.callbacks import Callback

from keras.layers import Dense, Dropout, Reshape, Conv1D, BatchNormalization, Activation, AveragePooling1D, GlobalAveragePooling1D, Lambda, Input, Concatenate, Add, UpSampling1D, Multiply

from sklearn.metrics import cohen_kappa_score, f1_score

from sklearn.model_selection import KFold, train_test_split





from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout
train = pd.read_csv("../input/liverpool-ion-switching/train.csv")

test = pd.read_csv("../input/liverpool-ion-switching/test.csv")

def normalize(df):

    arr = df['signal'].values

    arr_mean = arr.mean()

    arr_std = arr.std()

    arr = (arr - arr_mean)/arr_std

    df['signal'] = pd.DataFrame(arr)

    return df





# train = normalize(train)

# test = normalize(test)



train.head()
def feature(df):

    df.index = (df.time*10000 - 1).values

    df['batch'] = df.index // 50000 

    df['mean'] = df.groupby('batch')['signal'].mean()

    df['median'] = df.groupby('batch')['signal'].median()

    df['max'] = df.groupby('batch')['signal'].max()

    df['min'] = df.groupby('batch')['signal'].min()

    df['std'] = df.groupby('batch')['signal'].std()

    df = df.replace([np.inf, -np.inf], np.nan)    

    df.fillna(0, inplace=True)

    return df
train = feature(train)

test = feature(test)

col = [c for c in train.columns if c not in ['time', 'open_channels']]




train_input = train[col].values.reshape(-1,4000,7)

# train_input_mean = train_input.mean()

# train_input_sigma = train_input.std()

# train_input = (train_input-train_input_mean)/train_input_sigma

test_input = test[col].values.reshape(-1,10000,7)#

# test_input = (test_input-train_input_mean)/train_input_sigma



#train_target = df_train["open_channels"].values.reshape(-1,4000,1)#regression

train_target = pd.get_dummies(train["open_channels"]).values.reshape(-1,4000,11)

train_input.shape
train_x,valid_x,train_y,valid_y = train_test_split(train_input,train_target,random_state = 111,test_size = 0.2)
print(train_x.shape)

print(valid_x.shape)

print(train_y.shape)

print(valid_y.shape)
class macroF1(Callback):

    def __init__(self, model, inputs, targets):

        self.model = model

        self.inputs = inputs

        self.targets = np.argmax(targets, axis=2).reshape(-1)



    def on_epoch_end(self, epoch, logs):

        pred = np.argmax(self.model.predict(self.inputs), axis=2).reshape(-1)

        f1_val = f1_score(self.targets, pred, average="macro")

        print("val_f1_macro_score: ", f1_val)

        

        

def augmentations(input_data, target_data):

    #flip

    if np.random.rand()<0.5:    

        input_data = input_data[::-1]

        target_data = target_data[::-1]



    return input_data, target_data





def Datagen(input_dataset, target_dataset, batch_size, is_train=False):

    x=[]

    y=[]

  

    count=0

    idx_1 = np.arange(len(input_dataset))

    np.random.shuffle(idx_1)



    while True:

        for i in range(len(input_dataset)):

            input_data = input_dataset[idx_1[i]]

            target_data = target_dataset[idx_1[i]]

            



            if is_train:

                input_data, target_data = augmentations(input_data, target_data)

                

                

            x.append(input_data)

            y.append(target_data)

            count+=1

            if count==batch_size:

                x=np.array(x, dtype=np.float32)

                y=np.array(y, dtype=np.float32)

                inputs = x

                targets = y       

                x = []

                y = []

                count=0

                yield inputs, targets
def cbr(x, out_layer, kernel, stride, dilation):

    x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    return x



def se_block(x_in, layer_n):

    x = GlobalAveragePooling1D()(x_in)

    x = Dense(layer_n//8, activation="relu")(x)

    x = Dense(layer_n, activation="sigmoid")(x)

    x_out=Multiply()([x_in, x])

    return x_out



def resblock(x_in, layer_n, kernel, dilation, use_se=True):

    x = cbr(x_in, layer_n, kernel, 1, dilation)

    x = cbr(x, layer_n, kernel, 1, dilation)

    if use_se:

        x = se_block(x, layer_n)

    x = Add()([x_in, x])

    return x  



def Unet(input_shape=(None,7)):

    layer_n = 128

    kernel_size = 7

    depth = 2



    input_layer = Input(input_shape)    

    input_layer_1 = AveragePooling1D(5)(input_layer)

    input_layer_2 = AveragePooling1D(25)(input_layer)

    

    ########## Encoder

    x = cbr(input_layer, layer_n, kernel_size, 1, 1)#1000

    for i in range(depth):

        x = resblock(x, layer_n, kernel_size, 1)

    out_0 = x



    x = cbr(x, layer_n*2, kernel_size, 5, 1)

    for i in range(depth):

        x = resblock(x, layer_n*2, kernel_size, 1)

    out_1 = x



    x = Concatenate()([x, input_layer_1])    

    x = cbr(x, layer_n*3, kernel_size, 5, 1)

    for i in range(depth):

        x = resblock(x, layer_n*3, kernel_size, 1)

    out_2 = x



    x = Concatenate()([x, input_layer_2])    

    x = cbr(x, layer_n*4, kernel_size, 5, 1)

    for i in range(depth):

        x = resblock(x, layer_n*4, kernel_size, 1)

    

    ########### Decoder

    x = UpSampling1D(5)(x)

    x = Concatenate()([x, out_2])

    x = cbr(x, layer_n*3, kernel_size, 1, 1)



    x = UpSampling1D(5)(x)

    x = Concatenate()([x, out_1])

    x = cbr(x, layer_n*2, kernel_size, 1, 1)



    x = UpSampling1D(5)(x)

    x = Concatenate()([x, out_0])

    x = cbr(x, layer_n, kernel_size, 1, 1)    



    #regressor

    #x = Conv1D(1, kernel_size=kernel_size, strides=1, padding="same")(x)

    #out = Activation("sigmoid")(x)

    #out = Lambda(lambda x: 12*x)(out)

    

    #classifier

    x = Conv1D(11, kernel_size=kernel_size, strides=1, padding="same")(x)

    out = Activation("softmax")(x)

    

    model = Model(input_layer, out)

    

    return model
model = Unet()

model.summary()
def lrs(epoch):

    if epoch<35:

        lr = learning_rate

    elif epoch<50:

        lr = learning_rate/10

    else:

        lr = learning_rate/100

    return lr





learning_rate=0.0015

n_epoch=100

batch_size=32



lr_schedule = LearningRateScheduler(lrs)



#regressor

#model.compile(loss="mean_squared_error", 

#              optimizer=Adam(lr=learni'ng_rate),

#              metrics=["mean_absolute_error"])



#classifier

model.compile(loss=categorical_crossentropy, 

              optimizer=Adam(lr=learning_rate), 

              metrics=["accuracy"])
# hist = model.fit_generator(Datagen(train_x, train_y, batch_size, is_train=True),

#                             steps_per_epoch = len(train_x) // batch_size,

#                             epochs = n_epoch,

#                             validation_data=Datagen(valid_x, valid_y, batch_size),

#                             validation_steps = len(valid_x) // batch_size,

#                             callbacks = [lr_schedule, macroF1(model, valid_x, valid_y)],

#                             shuffle = False,

#                             verbose = 1 )

    

pred = np.argmax((model.predict(valid_x)+model.predict(valid_x[:,::-1,:])[:,::-1,:])/2, axis=2).reshape(-1)

gt = np.argmax(valid_y, axis=2).reshape(-1)

print("SCORE_oldmetric: ", cohen_kappa_score(gt, pred, weights="quadratic"))

print("SCORE_newmetric: ", f1_score(gt, pred, average="macro"))
pred = np.argmax((model.predict(test_input)+model.predict(test_input[:,::-1,:])[:,::-1,:])/2, axis=2).reshape(-1)



df_sub = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv", dtype={'time':str})

df_sub.open_channels = np.array(np.round(pred,0), np.int)

df_sub.to_csv("submission.csv",index=False)
p =model.predict(test_input)

p2 = model.predict(test_input[:,::-1,:])
p2.shape