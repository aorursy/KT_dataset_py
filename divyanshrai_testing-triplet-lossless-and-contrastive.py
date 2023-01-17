import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

from keras.models import model_from_json

import cv2



os.listdir("../input")
gent="../input/sigcomp-2009/sigcomp 2009/genuines"

forgt="../input/sigcomp-2009/sigcomp 2009/forgeries"



base="../input/different-triplet-loss-function-on-signature/"

os.listdir(base)


json_file = open(base+'model_triplet_loss_architecture.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

print("Architecture loaded from disk")



model_triplet_loss = model_from_json(loaded_model_json)

# load weights into new model

model_triplet_loss.load_weights(base+"model_triplet_loss_weights.h5")

print("Loaded model from disk")
json_file = open(base+'model_lossless_triplet_loss_architecture.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

print("Architecture loaded from disk")



model_lossless_triplet_loss = model_from_json(loaded_model_json)

# load weights into new model

model_lossless_triplet_loss.load_weights(base+"model_lossless_triplet_loss_weights.h5")

print("Loaded model from disk")


json_file = open(base+'model_contrastive_loss_architecture.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

print("Architecture loaded from disk")



model_contrastive_loss = model_from_json(loaded_model_json)

# load weights into new model

model_contrastive_loss.load_weights(base+"model_contrastive_loss_weights.h5")

print("Loaded model from disk")
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
tneg,tpos=0,0

list_with_existing_negs=[0.002, 0.008, 0.016, 0.018, 0.024, 0.033, 0.035, 0.044, 0.046, 0.063,

   0.070, 0.071, 0.077, 0.084, 0.085, 0.086, 0.089, 0.092, 0.093]

for k in list_with_existing_negs: #the id of signatures you want to check

    #print("When k is ", k)

    anc,neg,pos=getfiles(k,gent,forgt)

    tneg=tneg+len(neg)

    tpos=tpos+len(pos)

print(tneg,tpos)
forg_passed=[0,0,0]

gen_flagged=[0,0,0]

thresh=32     #hyperparameter

for k in list_with_existing_negs: #the id of signatures you want to check

    print("When k is ", k)

    anc,neg,pos=getfiles(k,gent,forgt)

    

    anchor=returnimages(gent,anc)

    for i in range(len(pos)): #pos

        positive=returnimages(gent,pos[i])

        x1=model_triplet_loss.predict([anchor,positive,anchor])

        x2=model_lossless_triplet_loss.predict([anchor,positive,anchor])

        x3=model_contrastive_loss.predict([anchor,positive,anchor])

        

        a1, p1, useless = x1[0,0:512], x1[0,512:1024], x1[0,1024:1536]

        a2, p2, useless = x2[0,0:512], x2[0,512:1024], x2[0,1024:1536]

        a3, p3, useless = x3[0,0:512], x3[0,512:1024], x3[0,1024:1536]

        

        dist1=np.linalg.norm(a1-p1)

        dist2=np.linalg.norm(a2-p2)

        dist3=np.linalg.norm(a3-p3)

        #print("positive distance is ",dist)

        

        if(dist1>thresh):

            gen_flagged[0]=gen_flagged[0]+1

        if(dist2>thresh):

            gen_flagged[1]=gen_flagged[1]+1

        if(dist3>18):

            gen_flagged[2]=gen_flagged[2]+1

     #       print("gen flagged - ",dist, "file name is - ", pos[i])

        #   print("1")



        

    for j in range(len(neg)): #neg

        negative=returnimages(forgt,neg[j])

        x1=model_triplet_loss.predict([anchor,negative,anchor])

        x2=model_lossless_triplet_loss.predict([anchor,negative,anchor])

        x3=model_contrastive_loss.predict([anchor,negative,anchor])

        

        a1, n1, useless = x1[0,0:512], x1[0,512:1024], x1[0,1024:1536]

        a2, n2, useless = x2[0,0:512], x2[0,512:1024], x2[0,1024:1536]

        a3, n3, useless = x3[0,0:512], x3[0,512:1024], x3[0,1024:1536]

        #dist=sum(a-n)

        

        dist1=np.linalg.norm(a1-n1)

        dist2=np.linalg.norm(a2-n2)

        dist3=np.linalg.norm(a3-n3)

        

        #print("negative distance is ",dist)

        if(dist1<=thresh):

            forg_passed[0]=forg_passed[0]+1

        if(dist2<=thresh):

            forg_passed[1]=forg_passed[1]+1

        if(dist3<=18):

            forg_passed[2]=forg_passed[2]+1
print("forg_passed by model_triplet_loss is ",forg_passed[0])

print("gen_flagged by model_triplet_loss is ",gen_flagged[0])

print("forg_passed by model_lossless_triplet_loss is ",forg_passed[1])

print("gen_flagged by model_lossless_triplet_loss is ",gen_flagged[1])

print("forg_passed by model_contrastive_loss is ",forg_passed[2])

print("gen_flagged by model_contrastive_loss is ",gen_flagged[2])
triplet_loss_TP=tpos-gen_flagged[0]

triplet_loss_FP=gen_flagged[0]

triplet_loss_TN=tneg-forg_passed[0]

triplet_loss_FN=forg_passed[0]



triplet_loss_precision=triplet_loss_TP/(triplet_loss_TP+triplet_loss_FP)

triplet_loss_recall=triplet_loss_TP/(triplet_loss_TP+triplet_loss_FN)

triplet_loss_f1score=2*((triplet_loss_precision*triplet_loss_recall)/(triplet_loss_precision+triplet_loss_recall))



lossless_triplet_loss_TP=tpos-gen_flagged[1]

lossless_triplet_loss_FP=gen_flagged[1]

lossless_triplet_loss_TN=tneg-forg_passed[1]

lossless_triplet_loss_FN=forg_passed[1]



lossless_triplet_loss_precision=lossless_triplet_loss_TP/(lossless_triplet_loss_TP+lossless_triplet_loss_FP)

lossless_triplet_loss_recall=lossless_triplet_loss_TP/(lossless_triplet_loss_TP+lossless_triplet_loss_FN)

lossless_triplet_loss_f1score=2*((lossless_triplet_loss_precision*lossless_triplet_loss_recall)/(lossless_triplet_loss_precision+lossless_triplet_loss_recall))



contrastive_loss_TP=tpos-gen_flagged[2]

contrastive_loss_FP=gen_flagged[2]

contrastive_loss_TN=tneg-forg_passed[2]

contrastive_loss_FN=forg_passed[2]



contrastive_loss_precision=contrastive_loss_TP/(contrastive_loss_TP+contrastive_loss_FP)

contrastive_loss_recall=contrastive_loss_TP/(contrastive_loss_TP+contrastive_loss_FN)

contrastive_loss_f1score=2*((contrastive_loss_precision*contrastive_loss_recall)/(contrastive_loss_precision+contrastive_loss_recall))
print("The triplet loss has precision, recall and f1 score as               " +str(triplet_loss_precision)[:7]+", "+str(triplet_loss_recall)[:7]+", "+str(triplet_loss_f1score)[:7])

print("The lossless triplet loss has precision, recall and f1 score as      " +str(lossless_triplet_loss_precision)[:7]+", "+str(lossless_triplet_loss_recall)[:7]+", "+str(lossless_triplet_loss_f1score)[:7])

print("The contrastive loss has precision, recall and f1 score as           " +str(contrastive_loss_precision)[:7]+", "+str(contrastive_loss_recall)[:7]+", "+str(contrastive_loss_f1score)[:7])
triplet_loss_recall