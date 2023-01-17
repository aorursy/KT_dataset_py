import os 

import cv2 

import matplotlib.pyplot as pl

import numpy as np

import random

#Neural Network Material

from tensorflow.keras.models import Sequential,Model

from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,AveragePooling2D,Embedding

from tensorflow.keras.layers import BatchNormalization,Lambda,Input,Layer,concatenate,Activation

from tensorflow.keras.regularizers import l2

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.applications import resnet50

import tensorflow.keras.backend as K

import tensorflow as tf

import tensorflow.keras as keras
img = cv2.imread('/kaggle/input/omniglot/images_background/Gujarati/character02/0419_05.png')

pl.imshow(img)

pl.show()
def loadImages(path):

    X = []

    alpha = 0

    chars = 0

    for a in os.listdir(path):

        alpha += 1

        for b in os.listdir(os.path.join(path,a)):

            chars += 1

            X.append([alpha,chars,cv2.imread(os.path.join(path,a,b),0) // 255])

    

    X = np.array(X)

    return X

path = '/kaggle/input/omniglot/images_background/Gujarati'

X = loadImages(path)

path = '/kaggle/input/omniglot/images_background/Greek'

XNew=loadImages(path)
print(X.shape)

print(X[959][0])

print(X[959][1])

print(X[:,2][:].shape)

print()

print(len(XNew)-1)

print(XNew.shape[0])

X[:,2:3] = X[:,2:3]
def randomizer(s='train'):

    if s == 'train':

        x = X

    else:

        x = XNew

    Pair = [[],[],[]]

    Labels = []

    for b in range(0,x.shape[0]-22,5):

        Pair[0].append(x[b][2])

        Pair[1].append(x[b+1][2])

        while True:

            cl = x[len(x)-1][0]

            r=random.randint(0,cl-1)

            if b!=r:

                break

        Pair[2].append(x[b+20][2])

    Pair = np.array(Pair)

    Pair.shape = Pair.shape[0],Pair.shape[1],Pair.shape[2],Pair.shape[3],1

    return Pair

def generator(batch,s='train'):

    for _ in range(batch):

        while True:

            pairs = randomizer(s)

        

            #print(pairs.shape)

            #print(targets.shape)

            yield (pairs[0][:],pairs[1][:],pairs[2][:]),np.array([0])
Pair = randomizer()

Pair.shape = Pair.shape[0],Pair.shape[1],Pair.shape[2],Pair.shape[3]

print(Pair[0][0].shape)

for a in range(10):

    f,ax = pl.subplots(1,3)

    ax[0].imshow(Pair[0][a])

    ax[1].imshow(Pair[1][a])

    ax[2].imshow(Pair[2][a])

    f.show()



#Pair = Pair.reshape(Pair.shape[0], Pair.shape[1],Pair.shape[2],Pair.shape[3],1)

print(Pair.shape)
def resnet_layer(inputs,

                 num_filters=16,

                 kernel_size=3,

                 strides=1,

                 activation='relu',

                 batch_normalization=True,

                 conv_first=True):

    conv = Conv2D(num_filters,

                  kernel_size=kernel_size,

                  strides=strides,

                  padding='same',

                  kernel_initializer='he_normal',

                  kernel_regularizer=l2(1e-4))



    x = inputs

    if conv_first:

        x = conv(x)

        if batch_normalization:

            x = BatchNormalization()(x)

        if activation is not None:

            x = Activation(activation)(x)

    else:

        if batch_normalization:

            x = BatchNormalization()(x)

        if activation is not None:

            x = Activation(activation)(x)

        x = conv(x)

    return x





def resnet_v1(input_shape,vals, depth=32, num_classes=10):

    if (depth - 2) % 6 != 0:

        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

    # Start model definition.

    num_filters = 16

    num_res_blocks = int((depth - 2) / 6)



    inputs = Input(shape=input_shape)

    x = resnet_layer(inputs=inputs)

    # Instantiate the stack of residual units

    for stack in range(3):

        for res_block in range(num_res_blocks):

            strides = 1

            if stack > 0 and res_block == 0:  # first layer but not first stack

                strides = 2  # downsample

            y = resnet_layer(inputs=x,

                             num_filters=num_filters,

                             strides=strides)

            y = resnet_layer(inputs=y,

                             num_filters=num_filters,

                             activation=None)

            if stack > 0 and res_block == 0:  # first layer but not first stack

                # linear projection residual shortcut connection to match

                # changed dims

                x = resnet_layer(inputs=x,

                                 num_filters=num_filters,

                                 kernel_size=1,

                                 strides=strides,

                                 activation=None,

                                 batch_normalization=False)

            x = keras.layers.add([x, y])

            x = Activation('relu')(x)

        num_filters *= 2



    # Add classifier on top.

    # v1 does not use BN after last shortcut connection-ReLU

    x = AveragePooling2D(pool_size=8)(x)

    y = Flatten()(x)

    dance = Dense(128,activation="sigmoid",kernel_initializer='he_normal',kernel_regularizer=l2(1e-4),bias_initializer="he_normal")(y)

    outputs = Lambda(lambda x:K.l2_normalize(x))(dance)

    model = Model(inputs=inputs, outputs=outputs)

    return model(vals)
Pair.shape = Pair.shape[0],Pair.shape[1],Pair.shape[2],Pair.shape[3],1



def getModel():

    #inp = Input(shape=INPUT)

    ashape = Input(Pair[0][0][:].shape)

    pshape = Input(Pair[1][0][:].shape)

    nshape = Input(Pair[2][0][:].shape)

        

    A = resnet_v1(input_shape=Pair[0][0][:].shape,vals=ashape)

    P = resnet_v1(input_shape=Pair[1][0][:].shape,vals=pshape)

    N = resnet_v1(input_shape=Pair[2][0][:].shape,vals=nshape)

    

    #triplet_loss_layer = TripletLossLayer(alpha=0.5, name='triplet_loss_layer')([A,P,N])

    #lam = Lambda(triplet_loss_dist)([A,P,N])

    #concatenate([A, P, N], axis=-1, name='merged_layer')    

    merged_vector = tf.keras.layers.concatenate([A, P, N], name='merged_layer')

    root = Model(inputs=[ashape, pshape, nshape], outputs=merged_vector)

    return root



print('Modelling Started')

root = getModel()

root.summary()

print('Modelling Finished')
root.layers[4].summary()
def triplet_loss(y_pred, alpha = 0.0,N=128):

    anchor = y_pred[:,0:N]

    positive = y_pred[:,N:N*2]

    negative = y_pred[:,N*2:N*3]

    pos_dist = tf.dtypes.cast(K.sum(K.square(anchor-positive),axis=1),tf.float32)

    neg_dist = tf.dtypes.cast(K.sum(K.square(anchor-negative),axis=1),tf.float32) 

    basic_loss = tf.subtract(pos_dist,neg_dist) + alpha

    loss = K.maximum(basic_loss,0.0)

    return loss

print('Compilation Started')

try:

    ad = tf.keras.optimizers.Adam(0.001)

    root.compile(loss=triplet_loss,optimizer=ad)

except Exception as A:

    print("Compilation Crash",A)

print('Compilation Finished')

print('Training Started')

RedLight = EarlyStopping(patience=2,monitor="loss",restore_best_weights=True,mode='auto')

RedLight1 = EarlyStopping(patience=2,monitor="val_loss",restore_best_weights=True,mode='auto')



History = root.fit_generator(generator(60),steps_per_epoch=20,epochs=200,validation_data=(generator(10,s="Val"))

                             ,validation_steps=10,use_multiprocessing=True,callbacks=[RedLight,RedLight1])

print('Training Finished')

#print('Gujarati Learning')

#root.evaluate([Pair[0][:],Pair[1][:]],Sim)

#print('Greek Learning')

#evaluate()
root.save_weights("TripletNet_Weights_OMNIGLOT.h5")

root.save("TripletNet_OMNIGLOT.h5")
import os

from IPython.display import FileLink

wname="TripletNet_Weights_OMNIGLOT.h5"

mname="TripletNet_OMNIGLOT.h5"

os.chdir('/kaggle/working')

FileLink(mname)
def show_analytics():

    pl.plot(History.history['loss'],c='r')

    pl.plot(History.history['val_loss'],'--',c='g')

    pl.title('Losses')

    pl.show()

show_analytics()
fig,ax = pl.subplots(1,3)

ax[0].imshow(X[random.randint(0,959)][2])

ax[1].imshow(X[random.randint(0,959)][2])

ax[2].imshow(X[random.randint(0,959)][2])

fig.show()
P = randomizer(s="valid")

P = P.reshape(P.shape[0],P.shape[1],P.shape[2],P.shape[3],1)

print(P.shape)



anchor = pred[:,0:128]

pos = pred[:,128:128*2]

neg = pred[:,128*2:128*3]



#anchor_dist = np.sum(anchor[:])



def triplet_dist(y_pred, alpha = 0.0,N=128):

    anchor = y_pred[:,0:N]

    positive = y_pred[:,N:N*2]

    negative = y_pred[:,N*2:N*3]

    pos_dist = np.sum(np.square(anchor-positive),axis=1)

    print("positive_distance : ",pos_dist)

    neg_dist =np.sum(np.square(anchor-negative),axis=1)

    print("negative_distance : ",neg_dist)

    basic_loss = np.subtract(pos_dist,neg_dist) + alpha

    print(basic_loss)

    loss = np.maximum(basic_loss,0.0)

    return loss

print(X[:][2].shape)

X1 = X[:][2][:]

for a in range(10):

    a,p,n = X1[random.randint(0,959)][2], X1[random.randint(0,959)][2], X1[random.randint(0,959)][2]

    print(triplet_dist(root.predict([[a],[p],[n]])))

    



pl.scatter([_ for _ in range(128)],anchor - anchor,c='b')

pl.scatter([_ for _ in range(128)],anchor - pos,c='r')

pl.scatter([_ for _ in range(128)],anchor - neg,c='g')

pl.plot(anchor)

pl.show()



from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

#%matplotlib notebook



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x =[1,2,3,4,5,6,7,8,9,10]

y =[5,6,2,3,13,4,1,2,4,8]

z =[2,3,3,3,5,7,9,11,9,10]



ax.scatter(anchor, pos, neg, c='r', marker='o')



ax.set_xlabel('X Label')

ax.set_ylabel('Y Label')

ax.set_zlabel('Z Label')



plt.show()
rpath = "/kaggle/input/tripletlossmodel/TripletNet_Margin_0.4.h5"

wpath = '/kaggle/input/tripletlossmodel/TripletNet_Weights_Margin_0.4.h5'

def triplet_loss(y_pred, alpha = 0,N = 128):

    anchor = y_pred[:,0:N]

    positive = y_pred[:,N:N*2]

    negative = y_pred[:,N*2:N*3]



    # distance between the anchor and the positive

    pos_dist = K.sum(K.square(anchor-positive),axis=1)



    # distance between the anchor and the negative

    neg_dist = K.sum(K.square(anchor-negative),axis=1)



    # compute loss

    basic_loss = pos_dist-neg_dist+alpha

    loss = K.maximum(basic_loss,0.0)

 

    return loss

root = tf.keras.models.load_model(rpath,custom_objects={'triplet_loss':triplet_loss})

root.load_weights(wpath)

#root = 
#print(X[0][2][:].reshape(105,105,1))

XP = []

for _ in range(len(X)):

    XP.append(X[_][2][:])

xp = random

triple=np.array(triple)

#scatter(triple,label)