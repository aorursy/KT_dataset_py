

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

from skimage.transform import resize

import time

import gc

import math

from sklearn.metrics import confusion_matrix



import os
import tensorflow as tf

from tensorflow.keras.layers import Activation, Input, Add, Dense, Concatenate, Conv2D, MaxPooling2D, GlobalMaxPooling2D

from tensorflow.keras.layers import BatchNormalization, Flatten, Dropout

from tensorflow.keras.initializers import glorot_uniform

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras import optimizers

from tensorflow.keras  import layers
IM_SIZE = 64

val_split = 0.2

train_rate = 1 - val_split

n_file = 4

n_each_train = 50210

n2_train = 12000
def conv2d_unit(array, conv, stride, padding = True):

    #array.shape = (x,y)

    #conv.shape = (x,y)

    size = conv.shape[0]

    pad = int(size/2)

    L1 = array.shape[0]

    L3 = int(math.ceil(L1/stride))



    if padding == True:

        L2 = L1 + 2*pad 

        

        array2 = np.zeros((L2, L2))

        array3 = np.zeros((L3, L3))

        array2[pad:L1+pad, pad:L1+pad] = array[:,:]

        

        for i in range(L3):

            for j in range(L3):

                x2 = stride*i

                y2 = stride*j

                a = np.sum(array2[x2:x2+size,y2:y2+size]*conv)

                

                array3[i,j] = a

        

        return array3



def conv2d(array, conv, stride = 1, padding = True):

    #array.shape = (x,y,n)

    #conv.shape =(x,y,c,n)

    n1 = array.shape[2]

    n2 = conv.shape[3]

    L1 = array.shape[0]

    L3 = int(math.ceil(L1/stride))    

    

    array3 = np.zeros((L3,L3,n2))

    for i in range(n2):

        for j in range(n1):

            array3[:,:,i] += conv2d_unit(array[:,:,j], conv[:,:,0,i], stride)

    

    return array3
def pooling_unit(array, size, stride):

    L1 = array.shape[0]

    L3 = int(math.ceil(L1/stride))

    pad = int(size/2)

    L2 = size + int(stride*L3*(L3-1)/2) + 2*pad



    array2 = np.zeros((L2, L2))    

    array3 = np.zeros((L3, L3))

    array2[pad:L1+pad,pad:L1+pad] = array[:,:]

    

    for i in range(L3):

        for j in range(L3):

            array3[i,j] = np.max(array2[stride*i:stride*i+size,stride*j:stride*j+size])

    

    return array3



def pooling(array, size = 2, stride = 2):

    #array.shape = (x,y,z) 

    N = array.shape[2]

    L1 = array.shape[0]

    L3 = int(math.ceil(L1/stride))

    array2 = np.zeros((L3,L3,N))

    for i in range(N):

        array2[:,:,i] = pooling_unit(array[:,:,i], size, stride)

    

    return array2
def adding_b(array, b):

    n = b.shape[0]

    for i in range(n):

        array[:,:,i] += b[i]

    

    return array
#one hot encoding of train label

def one_hot_encoder(data):

    array = np.zeros((data.shape[0], data.max() + 1))

    

    for i in range(data.shape[0]):

        array[i,data[i]] = 1

        

    return array
def cnn_model(FS, F):

    Input_X = Input(shape=(IM_SIZE,IM_SIZE,1))

    #64

    X = Conv2D(F,  kernel_size = FS, strides=(1, 1),  padding='same',

                kernel_initializer = glorot_uniform(seed=1), activation = "relu")(Input_X)

    X = Conv2D(F,  kernel_size = FS, strides=(1, 1),  padding='same',

                kernel_initializer = glorot_uniform(seed=1), activation = "relu")(X)  

    X = Conv2D(F,  kernel_size = FS, strides=(1, 1),  padding='same',

                kernel_initializer = glorot_uniform(seed=1), activation = "relu")(X)  

    X = Conv2D(F,  kernel_size = FS, strides=(1, 1),  padding='same',

                kernel_initializer = glorot_uniform(seed=1), activation = "relu")(X)  

    X = MaxPooling2D(pool_size=(2, 2), strides=2)(X)    

    #32

    F = 2*F

    X = Conv2D(F,  kernel_size = FS, strides=(1, 1),  padding='same',

                kernel_initializer = glorot_uniform(seed=1), activation = "relu")(X)

    X = Conv2D(F,  kernel_size = FS, strides=(1, 1),  padding='same',

                kernel_initializer = glorot_uniform(seed=1), activation = "relu")(X)

    X = Conv2D(F,  kernel_size = FS, strides=(1, 1),  padding='same',

                kernel_initializer = glorot_uniform(seed=1), activation = "relu")(X)

    X = Conv2D(F,  kernel_size = FS, strides=(1, 1),  padding='same',

                kernel_initializer = glorot_uniform(seed=1), activation = "relu")(X)  

    X = MaxPooling2D(pool_size=(2, 2), strides=2)(X)

    

    #16

    F = 2*F

    FS = 3

    X = Conv2D(F,  kernel_size = FS, strides=(1, 1),  padding='same',

                kernel_initializer = glorot_uniform(seed=1), activation = "relu")(X)

    X = Conv2D(F,  kernel_size = FS, strides=(1, 1),  padding='same',

                kernel_initializer = glorot_uniform(seed=1), activation = "relu")(X)  

    X = Conv2D(F,  kernel_size = FS, strides=(1, 1),  padding='same',

                kernel_initializer = glorot_uniform(seed=1), activation = "relu")(X)

    X = Conv2D(F,  kernel_size = FS, strides=(1, 1),  padding='same',

                kernel_initializer = glorot_uniform(seed=1), activation = "relu")(X)

    X = MaxPooling2D(pool_size=(2, 2), strides=2)(X)  

    

    #8    

    F = 2*F

    X = Conv2D(F,  kernel_size = FS, strides=(1, 1),  padding='same',

                kernel_initializer = glorot_uniform(seed=1), activation = "relu")(X)

    X = Conv2D(F,  kernel_size = FS, strides=(1, 1),  padding='same',

                kernel_initializer = glorot_uniform(seed=1), activation = "relu")(X)  

    X = Conv2D(F,  kernel_size = FS, strides=(1, 1),  padding='same',

                kernel_initializer = glorot_uniform(seed=1), activation = "relu")(X)  

    X = Conv2D(F,  kernel_size = FS, strides=(1, 1),  padding='same',

                kernel_initializer = glorot_uniform(seed=1), activation = "relu")(X)  

    X = MaxPooling2D(pool_size=(2, 2), strides=2)(X)

    F = 2*F



    #4

    #flatten

    X = Flatten()(X)

    X = BatchNormalization()(X)

    X = Dropout(0.4)(X)

    X = Dense(F, activation='relu', kernel_initializer= glorot_uniform(seed=1))(X)

    X = Dropout(0.4)(X)

    X = Dense(F, activation='relu', kernel_initializer= glorot_uniform(seed=1))(X)

    X = Dropout(0.35)(X)    

   

    Xg = Dense(168, activation='softmax', kernel_initializer='glorot_uniform')(X)    

    Xv = Dense(11, activation='softmax', kernel_initializer='glorot_uniform')(X)

    Xc = Dense(7, activation='softmax', kernel_initializer='glorot_uniform')(X)

        

    model = Model(inputs = Input_X, outputs = [Xg, Xv, Xc])

    adam = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00005, amsgrad=False)

    model.compile(optimizer = adam, loss = "categorical_crossentropy", metrics = ["accuracy"])

    

    return model    
def train_model(Model, epochs):

    history = Model.fit(x = train_m, y = [out_g, out_v, out_c], sample_weight = [weight_g*2, weight_v, weight_c], 

          validation_split=val_split, epochs = epochs, batch_size=250)

    return history
def train_model_final(Model, epochs):

    history = Model.fit(x = train_m, y = [out_g,out_g,out_g, out_v, out_c], sample_weight = [weight_g, weight_g,weight_g, weight_v, weight_c], 

          validation_split=val_split, epochs = epochs, batch_size=250)

    return history
def train_model_big(Model, epochs):

    history = Model.fit(x = train_m, y = [out_g,out_g,out_g, out_v, out_c], sample_weight = [weight_g,weight_g,weight_g, weight_v, weight_c], 

          validation_split=val_split, epochs = epochs, batch_size=250)

    return history
def generate_filtered_image(array, model):

    #array.shape = (x,y,1)

    conv1 = model.get_weights()[0]

    b1 = model.get_weights()[1]

    conv2 = model.get_weights()[2]

    b2 = model.get_weights()[3]

    conv3 = model.get_weights()[4]

    b3 = model.get_weights()[5]

    conv4 = model.get_weights()[6]

    b4 = model.get_weights()[7]

    

    conv5 = model.get_weights()[8]

    b5 = model.get_weights()[9]

    conv6 = model.get_weights()[10]

    b6 = model.get_weights()[11]

    conv7 = model.get_weights()[12]

    b7 = model.get_weights()[13]

    conv8 = model.get_weights()[14]

    b8 = model.get_weights()[15]

    

    conv9 = model.get_weights()[16]

    b9 = model.get_weights()[17]

    conv10 = model.get_weights()[18]

    b10 = model.get_weights()[19]

    conv11 = model.get_weights()[20]

    b11 = model.get_weights()[21]

    conv12 = model.get_weights()[22]

    b12 = model.get_weights()[23]

    

    conv13 = model.get_weights()[24]

    b13 = model.get_weights()[25]

    conv14 = model.get_weights()[26]

    b14 = model.get_weights()[27]

    conv15 = model.get_weights()[28]

    b15 = model.get_weights()[29]    

    conv16 = model.get_weights()[30]

    b16 = model.get_weights()[31]       

    

    x1 = adding_b(conv2d(array, conv1), b1)

    x2 = adding_b(conv2d(x1, conv2), b2)

    x3 = adding_b(conv2d(x2, conv3), b3)

    x4 = adding_b(conv2d(x3, conv4), b4)

    xp1 = pooling(x4)

    

    x5 = adding_b(conv2d(xp1, conv5), b5)

    x6 = adding_b(conv2d(x5, conv6), b6)

    x7 = adding_b(conv2d(x6, conv7), b7)

    x8 = adding_b(conv2d(x7, conv8), b8)

    xp2 = pooling(x8)    

    

    x9 = adding_b(conv2d(xp2, conv9), b9)

    x10 = adding_b(conv2d(x9, conv10), b10)

    x11 = adding_b(conv2d(x10, conv11), b11)

    x12 = adding_b(conv2d(x11, conv12), b12)

    xp3 = pooling(x12)       

    

    x13 = adding_b(conv2d(xp3, conv13), b13)

    x14 = adding_b(conv2d(x13, conv14), b14)

    x15 = adding_b(conv2d(x14, conv15), b15)

    x16 = adding_b(conv2d(x15, conv16), b16)

    xp4 = pooling(x16)    

    

    return x1,x2,x3,x4,xp1, x5,x6,x7,x8,xp2, x9,x10,x11,x12,xp3, x13,x14,x15,x16,xp4
def generate_filtered_image_3(array, model, seq, stride):

    layer = []

    n = len(seq)

    m = len(model.get_weights())

    for i in range(m):

        layer.append(model.get_weights()[i])   

   

    X = []

    i = 0

    k = 0

    AR = array

    while k < n:

        

        if seq[k] == "c":

            x_tmp = conv2d(AR, layer[i], stride = stride[k])

            X.append(x_tmp)

            i += 1

            X.append(adding_b(x_tmp, layer[i]))

            i += 1

        elif seq[k] == "p":

            X.append(pooling(AR, stride = stride[k]))

        k += 1

        

        AR = X[-1]

        

    return X

    
def show_conv_images(array, name):

    n = array.shape[2]

    if n > 10:

        n=10 + int((n-10)/3)

    fig, ax = plt.subplots(1,n, figsize = (18,3))

    for i in range(n):

        ax[i].imshow(array[:,:,i], cmap = "gray")

        if i % 10 == 0:

            ax[i].set_title(name +" - "+ str(i+1))

        ax[i].set_xticks([], [])

        ax[i].set_yticks([], [])

    

    plt.show()
def show_original_image(array):

    fig, ax = plt.subplots(figsize = (3,3))

    ax.imshow(array, cmap = "gray")

    ax.set_title("original cropped image")

    ax.set_xticks([], [])

    ax.set_yticks([], [])

    plt.show()
class_map = pd.read_csv("/kaggle/input/bengaliai-cv19/class_map.csv")

train_label = pd.read_csv("/kaggle/input/bengaliai-cv19/train.csv")

test_label = pd.read_csv("/kaggle/input/bengaliai-cv19/test.csv")

sample_submission = pd.read_csv("/kaggle/input/bengaliai-cv19/sample_submission.csv")
train_m = pd.read_parquet("/kaggle/input/bengali-create-processed-file/processed_train.parquet")
train_m = np.reshape(train_m.values,(train_m.shape[0],64,64))
out_g = one_hot_encoder(train_label["grapheme_root"][0:train_m.shape[0]].values)

out_v = one_hot_encoder(train_label["vowel_diacritic"][0:train_m.shape[0]].values)

out_c = one_hot_encoder(train_label["consonant_diacritic"][0:train_m.shape[0]].values)
count_grapheme = train_label[["grapheme_root"]][0:train_m.shape[0]].groupby("grapheme_root").size()

max_graheme = count_grapheme.max()

weight_grapheme = pd.DataFrame({"grapheme_root": range(168), "W_g":max_graheme/(count_grapheme.values)})



count_vowel = train_label[["vowel_diacritic"]][0:train_m.shape[0]].groupby("vowel_diacritic").size()

max_vowel = count_vowel.max()

weight_vowel = pd.DataFrame({"vowel_diacritic": range(11), "W_v":max_vowel/(count_vowel.values)})



count_consonant = train_label[["consonant_diacritic"]][0:train_m.shape[0]].groupby("consonant_diacritic").size()

max_consonant = count_consonant.max()

weight_consonant = pd.DataFrame({"consonant_diacritic": range(7), "W_c":max_consonant/(count_consonant.values)})
weight_cosonant_dic = {}

for i in range(weight_consonant.shape[0]):

    key = weight_consonant.iloc[i,0]

    val = weight_consonant.iloc[i,1]

    weight_cosonant_dic[key] = val

    

weight_vowel_dic = {}

for i in range(weight_vowel.shape[0]):

    key = weight_vowel.iloc[i,0]

    val = weight_vowel.iloc[i,1]

    weight_vowel_dic[key] = val



weight_grapheme_dic = {}

weight_grapheme_dic2 = {}

weight_grapheme_dic3 = {}

for i in range(weight_grapheme.shape[0]):

    key = weight_grapheme.iloc[i,0]

    val = weight_grapheme.iloc[i,1]

    weight_grapheme_dic[key] = np.sqrt(2*val-1)

    weight_grapheme_dic2[key] = val

    weight_grapheme_dic3[key] = np.sqrt(2*val-1)*2
train_m = np.reshape(train_m, (train_m.shape[0], train_m.shape[1],train_m.shape[2], 1))
model = cnn_model(FS = 2, F = 16)
model.summary()
history_g = model.fit(x = train_m, y = [out_g, out_v, out_c], 

                    class_weight = [weight_grapheme_dic3,  

                                    weight_vowel_dic, weight_cosonant_dic], 

                    validation_split=val_split, 

          epochs = 10, 

          batch_size= 400, shuffle = True)
sample = 978

x1,x2,x3,x4,xp1, x5,x6,x7,x8,xp2, x9,x10,x11,x12,xp3, x13,x14,x15,x16,xp4 = generate_filtered_image(train_m[sample,:,:,:], model)
show_original_image(train_m[sample,:,:,0])

show_conv_images(x1, name = "conv1")

show_conv_images(x2, name = "conv2")

show_conv_images(x3, name = "conv3")

show_conv_images(x4, name = "conv4")

show_conv_images(xp1, name = "maxpool")



show_conv_images(x5, name = "conv5")

show_conv_images(x6, name = "conv6")

show_conv_images(x7, name = "conv7")

show_conv_images(x8, name = "conv8")

show_conv_images(xp2, name = "maxpool")



show_conv_images(x9, name = "conv9")

show_conv_images(x10, name = "conv10")

show_conv_images(x11, name = "conv11")

show_conv_images(x12, name = "conv12")

show_conv_images(xp3, name = "maxpool")



show_conv_images(x13, name = "conv13")

show_conv_images(x14, name = "conv14")

show_conv_images(x15, name = "conv15")

show_conv_images(x16, name = "conv16")

show_conv_images(xp4, name = "maxpool")
sample = 17001

x1,x2,x3,x4,xp1, x5,x6,x7,x8,xp2, x9,x10,x11,x12,xp3, x13,x14,x15,x16,xp4 = generate_filtered_image(train_m[sample,:,:,:], model)
show_original_image(train_m[sample,:,:,0])

show_conv_images(x1, name = "conv1")

show_conv_images(x2, name = "conv2")

show_conv_images(x3, name = "conv3")

show_conv_images(x4, name = "conv4")

show_conv_images(xp1, name = "maxpool")



show_conv_images(x5, name = "conv5")

show_conv_images(x6, name = "conv6")

show_conv_images(x7, name = "conv7")

show_conv_images(x8, name = "conv8")

show_conv_images(xp2, name = "maxpool")



show_conv_images(x9, name = "conv9")

show_conv_images(x10, name = "conv10")

show_conv_images(x11, name = "conv11")

show_conv_images(x12, name = "conv12")

show_conv_images(xp3, name = "maxpool")



show_conv_images(x13, name = "conv13")

show_conv_images(x14, name = "conv14")

show_conv_images(x15, name = "conv15")

show_conv_images(x16, name = "conv16")

show_conv_images(xp4, name = "maxpool")
model = cnn_model(FS = 3, F = 16)
model.summary()
history_g = model.fit(x = train_m, y = [out_g, out_v, out_c], 

                    class_weight = [weight_grapheme_dic3,  

                                    weight_vowel_dic, weight_cosonant_dic], 

                    validation_split=val_split, 

          epochs = 10, 

          batch_size= 400, shuffle = True)
sample = 978

x1,x2,x3,x4,xp1, x5,x6,x7,x8,xp2, x9,x10,x11,x12,xp3, x13,x14,x15,x16,xp4 = generate_filtered_image(train_m[sample,:,:,:], model)
show_original_image(train_m[sample,:,:,0])

show_conv_images(x1, name = "conv1")

show_conv_images(x2, name = "conv2")

show_conv_images(x3, name = "conv3")

show_conv_images(x4, name = "conv4")

show_conv_images(xp1, name = "maxpool")



show_conv_images(x5, name = "conv5")

show_conv_images(x6, name = "conv6")

show_conv_images(x7, name = "conv7")

show_conv_images(x8, name = "conv8")

show_conv_images(xp2, name = "maxpool")



show_conv_images(x9, name = "conv9")

show_conv_images(x10, name = "conv10")

show_conv_images(x11, name = "conv11")

show_conv_images(x12, name = "conv12")

show_conv_images(xp3, name = "maxpool")



show_conv_images(x13, name = "conv13")

show_conv_images(x14, name = "conv14")

show_conv_images(x15, name = "conv15")

show_conv_images(x16, name = "conv16")

show_conv_images(xp4, name = "maxpool")
sample = 17001

x1,x2,x3,x4,xp1, x5,x6,x7,x8,xp2, x9,x10,x11,x12,xp3, x13,x14,x15,x16,xp4 = generate_filtered_image(train_m[sample,:,:,:], model)
show_original_image(train_m[sample,:,:,0])

show_conv_images(x1, name = "conv1")

show_conv_images(x2, name = "conv2")

show_conv_images(x3, name = "conv3")

show_conv_images(x4, name = "conv4")

show_conv_images(xp1, name = "maxpool")



show_conv_images(x5, name = "conv5")

show_conv_images(x6, name = "conv6")

show_conv_images(x7, name = "conv7")

show_conv_images(x8, name = "conv8")

show_conv_images(xp2, name = "maxpool")



show_conv_images(x9, name = "conv9")

show_conv_images(x10, name = "conv10")

show_conv_images(x11, name = "conv11")

show_conv_images(x12, name = "conv12")

show_conv_images(xp3, name = "maxpool")



show_conv_images(x13, name = "conv13")

show_conv_images(x14, name = "conv14")

show_conv_images(x15, name = "conv15")

show_conv_images(x16, name = "conv16")

show_conv_images(xp4, name = "maxpool")
model = cnn_model(FS = 4, F = 16)
model.summary()
history_g = model.fit(x = train_m, y = [out_g, out_v, out_c], 

                    class_weight = [weight_grapheme_dic3,  

                                    weight_vowel_dic, weight_cosonant_dic], 

                    validation_split=val_split, 

          epochs = 10, 

          batch_size= 400, shuffle = True)
sample = 978

x1,x2,x3,x4,xp1, x5,x6,x7,x8,xp2, x9,x10,x11,x12,xp3, x13,x14,x15,x16,xp4 = generate_filtered_image(train_m[sample,:,:,:], model)
show_original_image(train_m[sample,:,:,0])

show_conv_images(x1, name = "conv1")

show_conv_images(x2, name = "conv2")

show_conv_images(x3, name = "conv3")

show_conv_images(x4, name = "conv4")

show_conv_images(xp1, name = "maxpool")



show_conv_images(x5, name = "conv5")

show_conv_images(x6, name = "conv6")

show_conv_images(x7, name = "conv7")

show_conv_images(x8, name = "conv8")

show_conv_images(xp2, name = "maxpool")



show_conv_images(x9, name = "conv9")

show_conv_images(x10, name = "conv10")

show_conv_images(x11, name = "conv11")

show_conv_images(x12, name = "conv12")

show_conv_images(xp3, name = "maxpool")



show_conv_images(x13, name = "conv13")

show_conv_images(x14, name = "conv14")

show_conv_images(x15, name = "conv15")

show_conv_images(x16, name = "conv16")

show_conv_images(xp4, name = "maxpool")
sample = 17001

x1,x2,x3,x4,xp1, x5,x6,x7,x8,xp2, x9,x10,x11,x12,xp3, x13,x14,x15,x16,xp4 = generate_filtered_image(train_m[sample,:,:,:], model)
show_original_image(train_m[sample,:,:,0])

show_conv_images(x1, name = "conv1")

show_conv_images(x2, name = "conv2")

show_conv_images(x3, name = "conv3")

show_conv_images(x4, name = "conv4")

show_conv_images(xp1, name = "maxpool")



show_conv_images(x5, name = "conv5")

show_conv_images(x6, name = "conv6")

show_conv_images(x7, name = "conv7")

show_conv_images(x8, name = "conv8")

show_conv_images(xp2, name = "maxpool")



show_conv_images(x9, name = "conv9")

show_conv_images(x10, name = "conv10")

show_conv_images(x11, name = "conv11")

show_conv_images(x12, name = "conv12")

show_conv_images(xp3, name = "maxpool")



show_conv_images(x13, name = "conv13")

show_conv_images(x14, name = "conv14")

show_conv_images(x15, name = "conv15")

show_conv_images(x16, name = "conv16")

show_conv_images(xp4, name = "maxpool")
model = cnn_model(FS = 5, F = 16)
model.summary()
history_g = model.fit(x = train_m, y = [out_g, out_v, out_c], 

                    class_weight = [weight_grapheme_dic3,  

                                    weight_vowel_dic, weight_cosonant_dic], 

                    validation_split=val_split, 

          epochs = 10, 

          batch_size= 400, shuffle = True)
sample = 978

x1,x2,x3,x4,xp1, x5,x6,x7,x8,xp2, x9,x10,x11,x12,xp3, x13,x14,x15,x16,xp4 = generate_filtered_image(train_m[sample,:,:,:], model)
show_original_image(train_m[sample,:,:,0])

show_conv_images(x1, name = "conv1")

show_conv_images(x2, name = "conv2")

show_conv_images(x3, name = "conv3")

show_conv_images(x4, name = "conv4")

show_conv_images(xp1, name = "maxpool")



show_conv_images(x5, name = "conv5")

show_conv_images(x6, name = "conv6")

show_conv_images(x7, name = "conv7")

show_conv_images(x8, name = "conv8")

show_conv_images(xp2, name = "maxpool")



show_conv_images(x9, name = "conv9")

show_conv_images(x10, name = "conv10")

show_conv_images(x11, name = "conv11")

show_conv_images(x12, name = "conv12")

show_conv_images(xp3, name = "maxpool")



show_conv_images(x13, name = "conv13")

show_conv_images(x14, name = "conv14")

show_conv_images(x15, name = "conv15")

show_conv_images(x16, name = "conv16")

show_conv_images(xp4, name = "maxpool")
sample = 17001

x1,x2,x3,x4,xp1, x5,x6,x7,x8,xp2, x9,x10,x11,x12,xp3, x13,x14,x15,x16,xp4 = generate_filtered_image(train_m[sample,:,:,:], model)
show_original_image(train_m[sample,:,:,0])

show_conv_images(x1, name = "conv1")

show_conv_images(x2, name = "conv2")

show_conv_images(x3, name = "conv3")

show_conv_images(x4, name = "conv4")

show_conv_images(xp1, name = "maxpool")



show_conv_images(x5, name = "conv5")

show_conv_images(x6, name = "conv6")

show_conv_images(x7, name = "conv7")

show_conv_images(x8, name = "conv8")

show_conv_images(xp2, name = "maxpool")



show_conv_images(x9, name = "conv9")

show_conv_images(x10, name = "conv10")

show_conv_images(x11, name = "conv11")

show_conv_images(x12, name = "conv12")

show_conv_images(xp3, name = "maxpool")



show_conv_images(x13, name = "conv13")

show_conv_images(x14, name = "conv14")

show_conv_images(x15, name = "conv15")

show_conv_images(x16, name = "conv16")

show_conv_images(xp4, name = "maxpool")
model = cnn_model(FS = 6, F = 16)
model.summary()
history_g = model.fit(x = train_m, y = [out_g, out_v, out_c], 

                    class_weight = [weight_grapheme_dic3,  

                                    weight_vowel_dic, weight_cosonant_dic], 

                    validation_split=val_split, 

          epochs = 10, 

          batch_size= 400, shuffle = True)
sample = 978

x1,x2,x3,x4,xp1, x5,x6,x7,x8,xp2, x9,x10,x11,x12,xp3, x13,x14,x15,x16,xp4 = generate_filtered_image(train_m[sample,:,:,:], model)
show_original_image(train_m[sample,:,:,0])

show_conv_images(x1, name = "conv1")

show_conv_images(x2, name = "conv2")

show_conv_images(x3, name = "conv3")

show_conv_images(x4, name = "conv4")

show_conv_images(xp1, name = "maxpool")



show_conv_images(x5, name = "conv5")

show_conv_images(x6, name = "conv6")

show_conv_images(x7, name = "conv7")

show_conv_images(x8, name = "conv8")

show_conv_images(xp2, name = "maxpool")



show_conv_images(x9, name = "conv9")

show_conv_images(x10, name = "conv10")

show_conv_images(x11, name = "conv11")

show_conv_images(x12, name = "conv12")

show_conv_images(xp3, name = "maxpool")



show_conv_images(x13, name = "conv13")

show_conv_images(x14, name = "conv14")

show_conv_images(x15, name = "conv15")

show_conv_images(x16, name = "conv16")

show_conv_images(xp4, name = "maxpool")
sample = 17001

x1,x2,x3,x4,xp1, x5,x6,x7,x8,xp2, x9,x10,x11,x12,xp3, x13,x14,x15,x16,xp4 = generate_filtered_image(train_m[sample,:,:,:], model)
show_original_image(train_m[sample,:,:,0])

show_conv_images(x1, name = "conv1")

show_conv_images(x2, name = "conv2")

show_conv_images(x3, name = "conv3")

show_conv_images(x4, name = "conv4")

show_conv_images(xp1, name = "maxpool")



show_conv_images(x5, name = "conv5")

show_conv_images(x6, name = "conv6")

show_conv_images(x7, name = "conv7")

show_conv_images(x8, name = "conv8")

show_conv_images(xp2, name = "maxpool")



show_conv_images(x9, name = "conv9")

show_conv_images(x10, name = "conv10")

show_conv_images(x11, name = "conv11")

show_conv_images(x12, name = "conv12")

show_conv_images(xp3, name = "maxpool")



show_conv_images(x13, name = "conv13")

show_conv_images(x14, name = "conv14")

show_conv_images(x15, name = "conv15")

show_conv_images(x16, name = "conv16")

show_conv_images(xp4, name = "maxpool")
model = cnn_model(FS = 7, F = 16)
model.summary()
history_g = model.fit(x = train_m, y = [out_g, out_v, out_c], 

                    class_weight = [weight_grapheme_dic3,  

                                    weight_vowel_dic, weight_cosonant_dic], 

                    validation_split=val_split, 

          epochs = 10, 

          batch_size= 400, shuffle = True)
sample = 978

x1,x2,x3,x4,xp1, x5,x6,x7,x8,xp2, x9,x10,x11,x12,xp3, x13,x14,x15,x16,xp4 = generate_filtered_image(train_m[sample,:,:,:], model)
show_original_image(train_m[sample,:,:,0])

show_conv_images(x1, name = "conv1")

show_conv_images(x2, name = "conv2")

show_conv_images(x3, name = "conv3")

show_conv_images(x4, name = "conv4")

show_conv_images(xp1, name = "maxpool")



show_conv_images(x5, name = "conv5")

show_conv_images(x6, name = "conv6")

show_conv_images(x7, name = "conv7")

show_conv_images(x8, name = "conv8")

show_conv_images(xp2, name = "maxpool")



show_conv_images(x9, name = "conv9")

show_conv_images(x10, name = "conv10")

show_conv_images(x11, name = "conv11")

show_conv_images(x12, name = "conv12")

show_conv_images(xp3, name = "maxpool")



show_conv_images(x13, name = "conv13")

show_conv_images(x14, name = "conv14")

show_conv_images(x15, name = "conv15")

show_conv_images(x16, name = "conv16")

show_conv_images(xp4, name = "maxpool")
sample = 17001

x1,x2,x3,x4,xp1, x5,x6,x7,x8,xp2, x9,x10,x11,x12,xp3, x13,x14,x15,x16,xp4 = generate_filtered_image(train_m[sample,:,:,:], model)
show_original_image(train_m[sample,:,:,0])

show_conv_images(x1, name = "conv1")

show_conv_images(x2, name = "conv2")

show_conv_images(x3, name = "conv3")

show_conv_images(x4, name = "conv4")

show_conv_images(xp1, name = "maxpool")



show_conv_images(x5, name = "conv5")

show_conv_images(x6, name = "conv6")

show_conv_images(x7, name = "conv7")

show_conv_images(x8, name = "conv8")

show_conv_images(xp2, name = "maxpool")



show_conv_images(x9, name = "conv9")

show_conv_images(x10, name = "conv10")

show_conv_images(x11, name = "conv11")

show_conv_images(x12, name = "conv12")

show_conv_images(xp3, name = "maxpool")



show_conv_images(x13, name = "conv13")

show_conv_images(x14, name = "conv14")

show_conv_images(x15, name = "conv15")

show_conv_images(x16, name = "conv16")

show_conv_images(xp4, name = "maxpool")