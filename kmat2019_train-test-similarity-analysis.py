import os

import matplotlib.pyplot as plt

import glob

import numpy as np

import pandas as pd

import tensorflow as tf

from keras.layers import Dense, Dropout, Reshape, Conv1D, BatchNormalization, Activation, AveragePooling1D, GlobalAveragePooling1D, Lambda, Input, Concatenate, Add, UpSampling1D, Multiply, MaxPooling1D

from keras.models import Model

from keras.objectives import mean_squared_error

from keras import backend as K

from keras.losses import binary_crossentropy, categorical_crossentropy

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler

from keras.initializers import random_normal

from keras.optimizers import Adam, RMSprop, SGD

from keras.callbacks import Callback



from sklearn.metrics import cohen_kappa_score, f1_score

from sklearn.model_selection import KFold, train_test_split



from keras.constraints import unit_norm

from sklearn.manifold import TSNE
df_train = pd.read_csv("../input/liverpool-ion-switching/train.csv")

df_test = pd.read_csv("../input/liverpool-ion-switching/test.csv")



train_input = df_train["signal"].values.reshape(-1,5000,1)#number_of_data:1000 x time_step:5000

train_input_mean = train_input.mean()

train_input_sigma = train_input.std()

train_input = (train_input-train_input_mean)/train_input_sigma

test_input = df_test["signal"].values.reshape(-1,5000,1)#

test_input = (test_input-train_input_mean)/train_input_sigma



#train_target = pd.get_dummies(df_train["open_channels"]).values.reshape(-1,5000,11)

train_target = np.array([[i]*int(train_input.shape[0]/10) for i in range(10)]).flatten()#batch class



idx = np.arange(train_input.shape[0])

train_idx, val_idx = train_test_split(idx, random_state = 111,test_size = 0.2, stratify = train_target)



val_input = train_input[val_idx]

train_input = train_input[train_idx] 

val_target = train_target[val_idx]

train_target = train_target[train_idx] 



print("train_input:{}, val_input:{}, train_target:{}, val_target:{}".format(train_input.shape, val_input.shape, train_target.shape, val_target.shape))
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



def Classifier(input_shape=(None,1)):

    layer_n = 96

    kernel_size = 7

    depth = 3



    input_layer = Input(input_shape)    

    input_layer_1 = AveragePooling1D(5)(input_layer)

    input_layer_2 = AveragePooling1D(25)(input_layer)

    

    x = cbr(input_layer, layer_n, kernel_size, 1, 1)

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

    

    x = GlobalAveragePooling1D()(x)

    x = Dense(128, activation="linear")(x)

    x = Lambda(lambda x: 3*x/tf.sqrt(tf.reduce_sum(x**2, axis=-1, keepdims=True)+1e-7))(x)#L2

    out = Dense(10, activation="softmax", kernel_constraint = unit_norm(), name="out")(x)



    model=Model(input_layer, out)

    

    return model



def augmentations(input_data, target_data):

    #flip

    if np.random.rand()<0.5:    

        input_data = input_data[::-1]

    return input_data, target_data



def Datagen(input_dataset, target_dataset, batch_size, is_train=False):

    x=[]

    y=[]

  

    count=0

    idx = np.arange(len(input_dataset))

    np.random.shuffle(idx)

    

    while True:

        for i in range(len(input_dataset)):

            input_data = input_dataset[idx[i]]

            target_data = target_dataset[idx[i]]



            if is_train:

                input_data, target_data = augmentations(input_data, target_data)



            x.append(input_data)

            y.append(target_data)



            count+=1

            if count==batch_size:

                x = np.array(x, dtype=np.float32)

                y = np.identity(10)[y].astype(np.float32)



                inputs = x

                targets = y

       

                x = []

                y = []

                count=0

                yield inputs, targets

    

def model_fit(model, train_inputs, train_targets, val_inputs, val_targets, n_epoch, batch_size=32):

    hist = model.fit_generator(

        Datagen(train_inputs, train_targets, batch_size, is_train=True),

        steps_per_epoch = len(train_inputs) // batch_size,

        epochs = n_epoch,

        validation_data=Datagen(val_inputs, val_targets, batch_size),

        validation_steps = len(val_inputs) // batch_size,

        callbacks = [lr_schedule],

        shuffle = False,

        verbose = 1

        )

    return hist



def lrs(epoch):

    if epoch<40:

        lr = learning_rate

    elif epoch<60:

        lr = learning_rate/10

    else:

        lr = learning_rate/100

    return lr    
K.clear_session()

model = Classifier()

#print(model.summary())



learning_rate=0.0075

n_epoch=80

batch_size=64



lr_schedule = LearningRateScheduler(lrs)



model.compile(loss=categorical_crossentropy, 

              optimizer=Adam(lr=learning_rate), 

              metrics=["accuracy"])



hist = model_fit(model, train_input, train_target, val_input, val_target, n_epoch, batch_size)
from sklearn.manifold import TSNE

def tSNE_visualization(model, train_inputs, train_batch, test_inputs, test_batch):

    inputs = np.concatenate((train_inputs,test_inputs), axis=0)

    latent_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    latent_features = latent_model.predict(inputs)

    

    tsne_model = TSNE(n_components=2, perplexity=30, n_iter=1000)

    pred = tsne_model.fit_transform(latent_features)

    train_pred = pred[:len(train_inputs)]

    test_pred = pred[len(train_inputs):]



    fig, ax = plt.subplots(2, 1, figsize=(7,14))

    cmap = plt.get_cmap("tab10")

    

    for i in range(10):

        ax[0].scatter(train_pred[train_batch==i,0], train_pred[train_batch==i,1], alpha=0.5, s=int(40), color=cmap(i), edgecolors=None, label="train_batch_No.{}".format(i))

        ax[0].legend(bbox_to_anchor=(1.4,1), loc="upper right", borderaxespad=0, fontsize=12)

        ax[0].set_title('t-SNE Latent space -TRAIN-', fontsize=20)

    

    for i in range(4):

        ax[1].scatter(test_pred[test_batch==i,0], test_pred[test_batch==i,1], alpha=0.5, s=int(40), color=cmap(i), edgecolors=None, label="test_batch_No.{}".format(i))

        ax[1].legend(bbox_to_anchor=(1.4,1), loc="upper right", borderaxespad=0, fontsize=12)

        ax[1].set_title('t-SNE Latent space -TEST-', fontsize=20)

    plt.show()





test_batch = np.array([[i]*int(500000/5000) for i in range(4)]).flatten()



tSNE_visualization(model, val_input, val_target, test_input, test_batch)

test_class = np.argmax(model.predict(test_input), axis=-1)



cmap = plt.get_cmap("tab10")

    

fig, ax = plt.subplots(figsize=(25,8))

for i in range(10):

    ax.plot(np.arange(i*500000,(i+1)*500000), df_train["signal"].values[i*500000:(i+1)*500000], color=cmap(i), label="train_batch_No.{}".format(i))

ax.set_xticks(np.arange(0,5000000,100000))

ax.set_xlim(0,5000000)

ax.grid()

ax.set_title('TRAIN Signal', fontsize=15)

plt.show()



fig, ax = plt.subplots(2, 1, figsize=(10,16))



for i in range(4):

    ax[0].plot(np.arange(i*500000,(i+1)*500000), df_test["signal"].values[i*500000:(i+1)*500000], color=cmap(i), label="train_batch_No.{}".format(i))

    ax[1].plot(np.arange(i*100,(i+1)*100), test_class[i*100:(i+1)*100], color=cmap(i), label="train_batch_No.{}".format(i))

ax[0].set_xticks(np.arange(0,2000000,100000))

ax[1].set_xticks(np.arange(0,400,20))

ax[0].set_xlim(0,2000000)

ax[1].set_xlim(0,400)

ax[0].grid()

ax[1].grid()

ax[0].set_title('TEST Signal', fontsize=15)

ax[1].set_title('TEST predicted Batch Class', fontsize=15)

plt.show()
def add_feature(x):

    x_ = np.roll(x[:,:,0],1)

    x_[:,0] = 0

    x_ = (x_[:,:,np.newaxis] - x) 

    x__ = np.roll(x[:,:,0],-1)

    x__[:,-1] = 0

    x__ = (x__[:,:,np.newaxis] - x) 

    x = np.concatenate((x,x_,x__), axis=-1)

    return x



train_input = df_train["signal"].values.reshape(-1,5000,1)

train_input = add_feature(train_input)

train_input_mean = np.mean(train_input.reshape(len(df_train),-1), axis=0).reshape(1,1,-1)

train_input_sigma =np.std(train_input.reshape(len(df_train),-1), axis=0).reshape(1,1,-1)

train_input = (train_input-train_input_mean)/train_input_sigma

train_target = pd.get_dummies(df_train["open_channels"]).values.reshape(-1,5000,11)



test_input = df_test["signal"].values.reshape(-1,10000,1)#I like 10000 because the test data changes in 10000 time steps at minimum.

test_input = add_feature(test_input)

test_input = (test_input-train_input_mean)/train_input_sigma



batch_number = np.array([[i]*int(500000/5000) for i in range(10)]).flatten()

idx = np.arange(train_input.shape[0])



# select No.0-7 batch

idx = [idx[i] for i in range(len(batch_number)) if batch_number[i] in [0,1,2,3,4,5,6,7]]



train_idx, val_idx = train_test_split(idx, random_state = 111,test_size = 0.2)

val_input = train_input[val_idx]

train_input = train_input[train_idx] 

val_target = train_target[val_idx]

train_target = train_target[train_idx] 



print("train_input:{}, val_input:{}, train_target:{}, val_target:{}".format(train_input.shape, val_input.shape, train_target.shape, val_target.shape))
def aggregation_block(x_shallow, x_deep, deep_ch, out_ch):

    x_deep = UpSampling1D(5)(x_deep)

    x_deep = Conv1D(deep_ch, kernel_size=7, strides=1, padding="same")(x_deep)

    x_deep = BatchNormalization()(x_deep)   

    x_deep = Activation("relu")(x_deep)

    x = Concatenate()([x_shallow, x_deep])

    x = Conv1D(out_ch, kernel_size=1, strides=1, padding="same")(x)

    x = BatchNormalization()(x)   

    x = Activation("relu")(x)

    return x



def aggregation(skip_connections,output_layer_n,name=""):

    skip_connections_1=[]

    n=len(skip_connections)

    m=0

    for i in range(n):

        x= cbr(skip_connections[i], output_layer_n, 1, 1, [1])

        skip_connections_1.append(x)

    x_0= cbr(skip_connections[0], output_layer_n, 1, 1, [1])

    x_0 = aggregation_block(x_0, skip_connections[1], output_layer_n, output_layer_n)

    x_1= cbr(skip_connections[1], output_layer_n, 1, 1, [1])

    x_1 = aggregation_block(x_1, skip_connections[2], output_layer_n, output_layer_n)

    x_0 = aggregation_block(x_0, x_1, output_layer_n, output_layer_n)    

    x_2= cbr(skip_connections[2], output_layer_n, 1, 1, [1])

    skip_connections_out=[x_0,x_1,x_2]

    return skip_connections_out



def MinPooling1D(x, stride):

    x = Lambda(lambda x: -x)(x)

    x = MaxPooling1D(stride)(x)

    x = Lambda(lambda x: -x)(x)

    return x

    

def min_max_average_pooling(x_in, stride):

    x_av = AveragePooling1D(stride)(x_in)

    x_max =  MaxPooling1D(stride)(x_in)

    x_min = MinPooling1D(x_in, stride)

    x_out = Concatenate()([x_av,x_max,x_min])

    return x_out



def Unet(input_shape=(None,3)):

    layer_n = int(64*1.5)

    kernel_size = 7

    depth = int(2*1.5)

    dilations = [1]

    skip_connections = []



    input_layer = Input(input_shape)

    

    input_layer_1 = min_max_average_pooling(input_layer, 5)

    input_layer_2 = min_max_average_pooling(input_layer, 25)

    input_layer_3 = min_max_average_pooling(input_layer, 125)

    



    x = cbr(input_layer, layer_n, kernel_size, 1, 1)#1000

    for i in range(depth):

        x = resblock(x, layer_n, kernel_size, dilations)

    out_0 = x

    skip_connections.append(x)



    x = cbr(x, layer_n*2, kernel_size, 5, 1)

    for i in range(depth):

        x = resblock(x, layer_n*2, kernel_size, dilations)

    out_1 = x

    skip_connections.append(x)



    x = Concatenate()([x, input_layer_1])    

    x = cbr(x, layer_n*3, kernel_size, 5, 1)

    for i in range(depth):

        x = resblock(x, layer_n*3, kernel_size, dilations)

    out_2 = x

    skip_connections.append(x)



    x = Concatenate()([x, input_layer_2])    

    x = cbr(x, layer_n*4, kernel_size, 5, 1)

    for i in range(depth):

        x = resblock(x, layer_n*4, kernel_size, dilations)



    x = Concatenate()([x, input_layer_3])

    x = cbr(x, layer_n*4, kernel_size, 1, 1)

    x = resblock(x, layer_n*4, kernel_size, dilations)    



    skip_connections=aggregation(skip_connections,layer_n)



    x = UpSampling1D(5)(x)

    x = Concatenate()([x, skip_connections[2]])

    x = cbr(x, layer_n*3, kernel_size, 1, 1)



    x = UpSampling1D(5)(x)

    x = Concatenate()([x, skip_connections[1]])

    x = cbr(x, layer_n*2, kernel_size, 1, 1)



    x = UpSampling1D(5)(x)

    x = Concatenate()([x, skip_connections[0]])

    x = cbr(x, layer_n, kernel_size, 1, 1)    



    x = Conv1D(11, kernel_size=kernel_size, strides=1, padding="same")(x)

    out = Activation("softmax", name="out")(x)

    

    model=Model(input_layer, out)

    

    return model



def augmentations(input_data, target_data):

    #flip

    if np.random.rand()<0.5:    

        input_data = input_data[::-1]

        target_data = target_data[::-1]

        

    return input_data, target_data



def Datagen(input_dataset, target_dataset, batch_size, is_train=False):

    x = []

    y = []

  

    count = 0

    idx = np.arange(len(input_dataset))

    np.random.shuffle(idx)



    while True:

        for i in range(len(input_dataset)):

            input_data = input_dataset[idx[i]]

            target_data = target_dataset[idx[i]]



            if is_train:

                input_data, target_data = augmentations(input_data, target_data)



            x.append(input_data)

            y.append(target_data)

            

            count += 1

            if count==batch_size:

                x = np.array(x, dtype=np.float32)

                y = np.array(y, dtype=np.float32)

                inputs = x

                targets = y

                

                x = []

                y = []

                count=0

                yield inputs, targets



class macroF1(Callback):

    def __init__(self, model, inputs, targets):

        self.model = model

        self.inputs = inputs

        self.targets = np.argmax(targets, axis=2).reshape(-1)



    def on_epoch_end(self, epoch, logs):

        pred = np.argmax(self.model.predict(self.inputs), axis=2).reshape(-1)

        f1_val = f1_score(self.targets, pred, average="macro")

        print("val_f1_macro_score: ", f1_val)

                

def model_fit(model, train_inputs, train_targets, val_inputs, val_targets, n_epoch, batch_size=32):

    hist = model.fit_generator(

        Datagen(train_inputs, train_targets, batch_size, is_train=True),

        steps_per_epoch = len(train_inputs) // batch_size,

        epochs = n_epoch,

        validation_data=Datagen(val_inputs, val_targets, batch_size),

        validation_steps = len(val_inputs) // batch_size,

        callbacks = [lr_schedule, macroF1(model, val_inputs, val_targets)],

        shuffle = False,

        verbose = 1

        )

    return hist



def cos_lrs(epoch):

    if epoch<200:

        lr = learning_rate - learning_rate*(1-np.cos(np.pi*epoch/200))/2 + learning_rate/200

    else:

        lr = learning_rate/200

    return lr

K.clear_session()

model = Unet()



n_epoch=230

batch_size=64

learning_rate=0.00075

lr_schedule = LearningRateScheduler(cos_lrs)



model.compile(loss=categorical_crossentropy, 

              optimizer=Adam(lr=learning_rate), 

              metrics=["accuracy"])



hist = model_fit(model, train_input, train_target, val_input, val_target, n_epoch, batch_size)
pred = np.argmax((model.predict(test_input)+model.predict(test_input[:,::-1,:])[:,::-1,:])/2, axis=2).reshape(-1)

df_sub = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv", dtype={'time':str})

df_sub.open_channels = np.array(np.round(pred,0), np.int)

df_sub.to_csv("submission.csv",index=False)

print(df_sub.head())