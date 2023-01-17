# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

os.listdir("/kaggle/input/")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras import applications

from keras.models import Model

from keras.layers import Input,concatenate

from keras.optimizers import Adam

import keras.backend as K

import tensorflow as tf

import cv2



import random
import tensorflow

print(tensorflow.__version__)
gen="../input/handwritten-signatures/sample_signature/sample_Signature/genuine"

forg="../input/handwritten-signatures/sample_signature/sample_Signature/forged"



gentr="../input/sigcomp-2009-train/sigcomp 2009 train/Sigcomp 2009 train/genuine"

forgtr="../input/sigcomp-2009-train/sigcomp 2009 train/Sigcomp 2009 train/forgeries"



gent="../input/sigcomp-2009/sigcomp 2009/genuines"

forgt="../input/sigcomp-2009/sigcomp 2009/forgeries"
img_width, img_height, channels = 224, 224, 3



dim = (img_width, img_height)



def to_rgb(img):

    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 

    img_rgb = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)

    return img_rgb



def returnimages(path,img):

    image=cv2.imread(path+"/"+ img)                  #bringing the image

    image=cv2.resize(image, (img_width, img_height))

    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image=to_rgb(image).reshape(1,img_width, img_height,3)/255.0       #resizing and normalizing    

    return image



def getfiles(num,gen,forg):

    a=os.listdir(gen)

    b=os.listdir(forg)

    c=str(num)

    c=c[2:]

    if(len(c)==2):

        c=c+"0"

    

    n,m=[],[]

    for i in b:

        if i.endswith(c+".png"):

            n=n+[i]

        elif i.endswith(c+".PNG"):

            n=n+[i]

    for i in a:

        if i.endswith(c+".png"):

            m=m+[i]

        elif i.endswith(c+".PNG"):

            m=m+[i]

    return m.pop(),n,m



def getfiles2(num):

    a=os.listdir(gentr)

    b=os.listdir(forgtr)

    c=str(num)

    c=c[2:]

    if(len(c)==2):

        c=c+"0"

    n,m=[],[]

    for i in b:

        if (i.endswith(c+"_001_6g.png") or i.endswith(c+"_002_6g.png") or i.endswith(c+"_003_6g.png")

            or i.endswith(c+"_004_6g.png") or i.endswith(c+"_005_6g.png")):

            n=n+[i]

        elif (i.endswith(c+"_001_6g.PNG") or i.endswith(c+"_002_6g.PNG") or i.endswith(c+"_003_6g.PNG")

              or i.endswith(c+"_004_6g.PNG") or i.endswith(c+"_005_6g.PNG")):

            n=n+[i]

    for i in a:

        if (i.endswith(c+"_001_6g.png") or i.endswith(c+"_002_6g.png") or i.endswith(c+"_003_6g.png")

            or i.endswith(c+"_004_6g.png") or i.endswith(c+"_005_6g.png")):

            m=m+[i]

        elif (i.endswith(c+"_001_6g.PNG") or i.endswith(c+"_002_6g.PNG") or i.endswith(c+"_003_6g.PNG")

              or i.endswith(c+"_004_6g.PNG") or i.endswith(c+"_005_6g.PNG")):

            m=m+[i]

    return m.pop(),n,m
def triplet_loss(y_true, y_pred):

    alpha = 0.5

    anchor, positive, negative =y_pred[0,0:512], y_pred[0,512:1024], y_pred[0,1024:1536]

    

    positive_distance = K.mean(K.square(anchor - positive),axis=-1)

    negative_distance = K.mean(K.square(anchor - negative),axis=-1)

    return K.mean(K.maximum(0.0, positive_distance - negative_distance + alpha))
def lossless_triplet_loss(y_true, y_pred, beta=3, epsilon=1e-8):

    anchor, positive, negative =y_pred[0,0:512], y_pred[0,512:1024], y_pred[0,1024:1536]

    

    pos_dist = K.mean(K.square(anchor - positive),axis=-1)

    neg_dist = K.mean(K.square(anchor - negative),axis=-1)

    

    N=3

    pos_dist = -tf.math.log(-tf.divide((pos_dist),beta)+1+epsilon)

    neg_dist = -tf.math.log(-tf.divide((N-neg_dist),beta)+1+epsilon)

    loss = neg_dist + pos_dist

    

    return loss



def contrastive_loss(y_true, y_pred):

    margin = 1

    square_pred = K.square(y_pred)

    margin_square = K.square(K.maximum(margin - y_pred, 0))

    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
def generator():

    for i in range(1,31):

        if(i<10):

            anc,neg,pos=getfiles(float("0.00"+str(i)),gen,forg)

        else:

            anc,neg,pos=getfiles(float("0.0"+str(i)),gen,forg)

        for i in range(len(neg)):

            for j in range(len(pos)):

                anchor=returnimages(gen,anc)

                positive=returnimages(gen,pos[j])

                negative=returnimages(forg,neg[i])

               # yield ([anc,pos[j],neg[i]],[0])

                yield ([anchor,positive,negative],[0])

                

def generator2():

    x=["0.001","0.004", "0.005", "0.006", "0.007","0.008", "0.009", "0.010", "0.011"]

    for k in x:

        anc,neg,pos=getfiles2(k)

        frac=0.95    

        inds = set(random.sample(list(range(len(neg))), int(frac*len(neg))))

        neg = [n for i,n in enumerate(neg) if i not in inds]

    

        for i in range(len(neg)):

            for j in range(len(pos)):

                anchor=returnimages(gentr,anc)

                positive=returnimages(gentr,pos[j])

                negative=returnimages(forgtr,neg[i])

               # yield ([anc,pos[j],neg[i]])

                yield ([anchor,positive,negative],[0])
model1 = applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='max')

for layer in model1.layers[:15]:

    layer.trainable = False



anchor_in = Input(shape=(img_width, img_height, channels))

pos_in = Input(shape=(img_width, img_height, channels))

neg_in = Input(shape=(img_width, img_height, channels))



anchor_out = model1(anchor_in)

pos_out = model1(pos_in)

neg_out = model1(neg_in)

merged_vector = concatenate([anchor_out, pos_out, neg_out],axis=1)



model_triplet_loss = Model(inputs=[anchor_in, pos_in, neg_in], outputs=merged_vector)

model_triplet_loss.compile(optimizer=Adam(lr=0.00001),loss=triplet_loss)
for x in range(1):

    model_triplet_loss.fit_generator(generator(),steps_per_epoch=100,epochs=6)

    

#for x in range(1):

#    model_contrastive_loss.fit_generator(generator2(),steps_per_epoch=32,epochs=9)
model2 = applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='max')

for layer in model2.layers[:15]:

    layer.trainable = False



anchor_in = Input(shape=(img_width, img_height, channels))

pos_in = Input(shape=(img_width, img_height, channels))

neg_in = Input(shape=(img_width, img_height, channels))



anchor_out = model2(anchor_in)

pos_out = model2(pos_in)

neg_out = model2(neg_in)

merged_vector = concatenate([anchor_out, pos_out, neg_out],axis=1)



model_lossless_triplet_loss = Model(inputs=[anchor_in, pos_in, neg_in], outputs=merged_vector)

model_lossless_triplet_loss.compile(optimizer=Adam(lr=0.0000019),loss=lossless_triplet_loss)
for x in range(1):

    model_lossless_triplet_loss.fit_generator(generator(),steps_per_epoch=100,epochs=6)

    

#for x in range(1):

#    model_lossless_triplet_loss.fit_generator(generator2(),steps_per_epoch=32,epochs=9)
model3 = applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='max')

for layer in model3.layers[:15]:

    layer.trainable = False



anchor_in = Input(shape=(img_width, img_height, channels))

pos_in = Input(shape=(img_width, img_height, channels))

neg_in = Input(shape=(img_width, img_height, channels))



anchor_out = model3(anchor_in)

pos_out = model3(pos_in)

neg_out = model3(neg_in)

merged_vector = concatenate([anchor_out, pos_out, neg_out],axis=1)



model_contrastive_loss = Model(inputs=[anchor_in, pos_in, neg_in], outputs=merged_vector)

model_contrastive_loss.compile(optimizer=Adam(lr=0.00001),loss=contrastive_loss)
for x in range(1):

    model_contrastive_loss.fit_generator(generator(),steps_per_epoch=100,epochs=6)



#for x in range(1):

#    model_triplet_loss.fit_generator(generator2(),steps_per_epoch=32,epochs=9)
tneg,tpos=0,0

x=[0.002, 0.008, 0.016, 0.018, 0.024, 0.033, 0.035, 0.044, 0.046, 0.063,

   0.070, 0.071, 0.077, 0.084, 0.085, 0.086, 0.089, 0.092, 0.093]

for k in x: #the id of signatures you want to check

    #print("When k is ", k)

    anc,neg,pos=getfiles(k,gent,forgt)

    tneg=tneg+len(neg)

    tpos=tpos+len(pos)

print(tneg,tpos)
# Save the weights

model_triplet_loss.save_weights('model_triplet_loss_weights.h5')

model_lossless_triplet_loss.save_weights('model_lossless_triplet_loss_weights.h5')

model_contrastive_loss.save_weights('model_contrastive_loss_weights.h5')



# Save the model architecture

with open('model_triplet_loss_architecture.json', 'w') as f:

    f.write(model_triplet_loss.to_json())

    

with open('model_lossless_triplet_loss_architecture.json', 'w') as f:

    f.write(model_lossless_triplet_loss.to_json())

    

with open('model_contrastive_loss_architecture.json', 'w') as f:

    f.write(model_contrastive_loss.to_json())