import os

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import cv2

import math



import keras

import keras.backend as K

import tensorflow as tf

from keras import applications

from keras.models import Model

from keras.layers import Flatten, Dense, Input,concatenate

from keras.optimizers import Adam

from keras.models import load_model, model_from_json





import random
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
model = applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='max')

for layer in model.layers[:15]:

    layer.trainable = False
anchor_in = Input(shape=(img_width, img_height, channels))

pos_in = Input(shape=(img_width, img_height, channels))

neg_in = Input(shape=(img_width, img_height, channels))



anchor_out = model(anchor_in)

pos_out = model(pos_in)

neg_out = model(neg_in)

merged_vector = concatenate([anchor_out, pos_out, neg_out],axis=1)



model = Model(inputs=[anchor_in, pos_in, neg_in], outputs=merged_vector)
os.listdir("../input/weights-for-signature-verification")
os.listdir("../input/training-using-greyscale")
#with open('../input/training-using-greyscale/model_architecture.json', 'r') as f:

#    model = model_from_json(f.read())



# Load weights into the new model

model.load_weights('../input/training-using-greyscale/model_weights.h5')
def getfilest(num,gen,forg):

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

    return m.pop(),m.pop(),m.pop(),n,m
#counting the number of forgeries and genuine in the test set

tneg,tpos=0,0

x=[0.002, 0.008, 0.016, 0.018, 0.024, 0.033, 0.035, 0.044, 0.046, 0.063,

   0.070, 0.071, 0.077, 0.084, 0.085, 0.086, 0.089, 0.092, 0.093]

for k in x: #the id of signatures you want to check

    #print("When k is ", k)

    anc1,anc2,anc3,neg,pos=getfilest(k,gent,forgt)

    tneg=tneg+len(neg)

    tpos=tpos+len(pos)

print(tneg,tpos)
forg_passed=0

gen_flagged=0

x=[0.002, 0.008, 0.016, 0.018, 0.024, 0.033, 0.035, 0.044, 0.046, 0.063,

   0.070, 0.071, 0.077, 0.084, 0.085, 0.086, 0.089, 0.092, 0.093]



for k in x: #the id of signatures you want to check

    print("When k is ", k)

    anc1,anc2,anc3,neg,pos=getfilest(k,gent,forgt)

    anchor1=returnimages(gent,anc1)

    anchor2=returnimages(gent,anc2)

    anchor3=returnimages(gent,anc3)

    

    x=model.predict([anchor1,anchor2,anchor3])

    

    a1, a2, a3 = x[0,0:511], x[0,512:1023], x[0,1024:1535]

    

    thresh1=np.linalg.norm(a1-a2)#+3.5

    thresh2=np.linalg.norm(a2-a3)

    thresh3=np.linalg.norm(a1-a3)

    thresh=(thresh1+thresh2+thresh3)/3

    thresh=thresh+3.5

    print("threshhold is  ",thresh)

    

    for i in range(len(pos)): #pos

        positive=returnimages(gent,pos[i])

        x=model.predict([anchor1,positive,anchor2])

        useless, p, useless = x[0,0:511], x[0,512:1023], x[0,1024:1535]

        dist1=np.linalg.norm(a1-p)

        dist2=np.linalg.norm(a2-p)

        dist3=np.linalg.norm(a3-p)

        dist=(dist1+dist2+dist3)/3

        

        

        if(dist>thresh):

        #  print("0")

            gen_flagged=gen_flagged+1

            print("gen flagged - ",dist1, "file name is - ", pos[i])

            

        else:

            gen_flagged=gen_flagged

        #   print("1")

    for j in range(len(neg)): #neg

        negative=returnimages(forgt,neg[j])

        x=model.predict([anchor1,negative,anchor2])

        useless, n, useless = x[0,0:511], x[0,512:1023], x[0,1024:1535]

        #dist=sum(a-n)

        dist1=np.linalg.norm(a1-n)

        dist2=np.linalg.norm(a2-n)

        dist3=np.linalg.norm(a3-n)

        #print("negative distance is ",dist)

        dist=(dist1+dist2+dist3)/3

        if(dist>thresh):

            forg_passed=forg_passed

          #  print("0")

        else:

            forg_passed=forg_passed+1

            print("forg passed - ",dist1, "file name is - ", neg[j])

          #  print("1")

        

print("forg_passed is ",forg_passed)

print("gen_flagged is ",gen_flagged)
#with the new weights its 

#38 - forgeries that go undetected out of 624

#27 - genuines that are flagged as forgeries out of 204