#!pip install keras-mxnet kerascv
#import  kerascv

#from kerascv.model_provider import get_model as kecv_get_model
#import gcimport gc

import os

import pickle

import random

import time

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import PIL

from PIL import Image, ImageOps

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

import keras

import matplotlib

import matplotlib.pyplot as plt

from keras.models import Model

import csv

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense

import tensorflow as tf

import time

from keras.engine.topology import Layer

import keras.backend as K
def constructpath(row,train=True):

    '''

    function to construct filepath for a single specific row

    '''

    ans=[]

    if train==True:

        path='../input/train/'

    else:

        path='../input/test/'

    experiment, plate, well =row['id_code'].split('_')

    path=path+str(experiment+'/'+'Plate'+plate+'/'+ well)

    for site in ['_s1','_s2']:

        for channel in ['_w1','_w2','_w3','_w4','_w5','_w6']:

            fp=path+site+channel+'.png'

            #if os.path.exists(fp):

            ans.append(fp)

    ans.sort()

    ans2=ans[6:]

    ans=ans[:6]

    return (tuple(ans),tuple(ans2))



def reduce_dims(batch, weights):

    return np.dot(batch,weights[:,:])



def picgenerator3D(

    trainDS2,Weights, basemodel,graph, site,batchsize=8, 

                 channelnumlist=[0,1,2,3,4,5],

                 shuffle=True,autoenc=False, 

                 test=False,

    ):

    trainDS=trainDS2.copy().reset_index(drop=True)

    shape=512

    try: shape=int(basemodel.input.get_shape()[1])

    except:shape=512

    while True:

        if shuffle==True:

            trainDS = trainDS.sample(frac=1).reset_index(drop=True)

        batch=[]

        ans=np.zeros((batchsize)).astype(int)

        for i, row in trainDS.iterrows():

            if test==False:

                ans[i%batchsize]=row['sirna']

            imageC=[]

            for channel in channelnumlist:

                imageC.append(Image.open(row['pathes'][site][channel]).resize((shape,shape)))

            batch.append(np.stack(imageC,axis=-1))

            if (i+1)%batchsize==0:

                with graph.as_default():

                    out=basemodel.predict(reduce_dims(np.stack(batch)/255, Weights))

                    out2=(np.quantile(out,0.9,axis=(1,2))+np.quantile(out,0.95,axis=(1,2)))/2

                if autoenc==True:

                    yield out2,out2

                else:

                    if test==False:

                        yield out2, ans

                    if test==True:

                        yield out2

                batch=[]

                ans=np.zeros((batchsize))

        if autoenc==True:

            with graph.as_default():

                out=basemodel.predict(reduce_dims(np.stack(batch)/255, Weights))

                out2=(np.quantile(out,0.9,axis=(1,2))+np.quantile(out,0.95,axis=(1,2)))/2

            yield out2,out2

        else:

            with graph.as_default():

                out=basemodel.predict(reduce_dims(np.stack(batch)/255, Weights))

                out2=(np.quantile(out,0.9,axis=(1,2))+np.quantile(out,0.95,axis=(1,2)))/2

            if test==False:

                yield out2, ans[:len(batch)]

            if test==True:

                yield out2

                

def Createmodel(basemodel, layers,reg=0.0,lr=0.001,regllast=0.0,arcface=False):

    shape=int(basemodel.output.shape[-1])

    inp=keras.layers.Input(shape=(shape,))

    label=keras.layers.Input(shape=(1108,))

    if layers[0]!=0:

        x=keras.layers.Dense(layers[0],activation='elu',kernel_regularizer=keras.regularizers.l2(reg),bias_regularizer=keras.regularizers.l2(reg))(inp)

        x=keras.layers.Dropout(0.2)(x)

        for obj in layers[1:]:

            x=keras.layers.Dense(obj,activation='relu',kernel_regularizer=keras.regularizers.l2(reg),bias_regularizer=keras.regularizers.l2(reg))(x)

            x=keras.layers.Dropout(0.2)(x)

    #x=keras.layers.Dense(64,activation='relu',kernel_regularizer=keras.regularizers.l2(0.0000),bias_regularizer=keras.regularizers.l2(0.00000))(x)

    #x=keras.layers.Dropout(01)(x)

    if layers[0]==0:

        x=keras.layers.Dropout(0.2)(inp)

    if arcface==True:out=ArcFace(1108, 30, 0.5)([x,label]) 

    if arcface==False:

        x=keras.layers.BatchNormalization()(x)

        x=keras.layers.Dense(1108,activation='elu',kernel_regularizer=keras.regularizers.l2(regllast),bias_regularizer=keras.regularizers.l2(regllast))(x)

        out=keras.layers.Activation('softmax')(x)

    if arcface==True:model=Model(input=[inp,label],outputs=out)

    if arcface==False:model=Model(input=inp,outputs=out)

    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='categorical_crossentropy',metrics=['categorical_accuracy'])

    return model



def featurecollector(DF, basemodel,weights, batchsize,site=1,test=False):

    inittime=time.time()

    generator=picgenerator3D(

    DF,weights,basemodel,graph,site,

    batchsize=BATCHSIZE,channelnumlist=[0,1,2,3,4,5], 

    autoenc=False,

    test=test,

    shuffle=False

        )

    toret=np.zeros((len(DF),int(basemodel.output.shape[-1])))

    ysez=np.zeros(len(DF))

    #print (toret.shape[0])

    for it in range(int(np.ceil(len(DF)/batchsize))):

        A=next(generator)

        if test==False:

            toret[it*batchsize:it*batchsize+A[0].shape[0],:]=A[0]

            ysez[it*batchsize:it*batchsize+A[1].shape[0]]=A[1]

            timefrombeg=time.time()-inittime

        else:

            toret[it*batchsize:it*batchsize+A.shape[0],:]=A

            #ysez[it*batchsize:it*batchsize+A[1].shape[0]]=A[1]

            timefrombeg=time.time()-inittime

        if it%30==0 and it>0: print ('remains', timefrombeg/it*(int(np.ceil(len(DF)/batchsize)-it)), ' s')

    return toret, ysez

def schedule(epoch,lr):

    return lr*0.9
from keras import backend as K

from keras.layers import Layer

from keras import regularizers



import tensorflow as tf

class ArcFace(Layer):

    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):

        super(ArcFace, self).__init__(**kwargs)

        self.n_classes = n_classes

        self.s = s

        self.m = m

        self.regularizer = regularizers.get(regularizer)



    def build(self, input_shape):

        super(ArcFace, self).build(input_shape[0])

        self.W = self.add_weight(name='W',

                                shape=(input_shape[0][-1], self.n_classes),

                                initializer='glorot_uniform',

                                trainable=True,

                                regularizer=self.regularizer)



    def call(self, inputs):

        x, y = inputs

        c = K.shape(x)[-1]

        # normalize feature

        x = tf.nn.l2_normalize(x, axis=1)

        # normalize weights

        W = tf.nn.l2_normalize(self.W, axis=0)

        # dot product

        logits = x @ W

        # add margin

        # clip logits to prevent zero division when backward

        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))

        target_logits = tf.cos(theta + self.m)

        # sin = tf.sqrt(1 - logits**2)

        # cos_m = tf.cos(logits)

        # sin_m = tf.sin(logits)

        # target_logits = logits * cos_m - sin * sin_m

        #

        logits = logits * (1 - y) + target_logits * y

        # feature re-scale

        logits *= self.s

        out = tf.nn.softmax(logits)



        return out



    def compute_output_shape(self, input_shape):

        return (None, self.n_classes)

weights=np.array([[ 0.36013198, -0.        ,  0.10044181],

       [ 0.04300603,  0.27274185,  0.3570403 ],

       [ 0.14618097,  0.20497076, -0.        ],

       [ 0.19643596,  0.14860706,  0.50436   ],

       [ 0.18471946,  0.22273754, -0.        ],

       [ 0.0695256 ,  0.15094277,  0.03815791]], dtype=float)
#Data reading and preprocessing

BATCHSIZE=32

DF=pd.read_csv('../input/recursion-cellular-image-classification/train.csv')

DF['pathes']=DF.apply(lambda x: constructpath(x),axis=1)

DF['celltype']=DF['experiment'].apply(lambda x: x.split('-')[0])

DF_train_control=pd.read_csv('../input/recursion-cellular-image-classification/train_controls.csv')

DF_train_control['pathes']=DF_train_control.apply(lambda x: constructpath(x),axis=1)

DF_test_control=pd.read_csv('../input/recursion-cellular-image-classification/test_controls.csv')

DF_test_control['pathes'] = DF_test_control.apply(lambda x: constructpath(x, train = False), axis = 1)



#get the samples by celltype

idxHEPG2=list(DF[DF['celltype']=='HEPG2'].index)

idxHUVEC=list(DF[DF['celltype']=='HUVEC'].index)

idxRPE=list(DF[DF['celltype']=='RPE'].index)

idxU2OS=list(DF[DF['celltype']=='U2OS'].index)



submittionDF=pd.read_csv('../input/recursion-cellular-image-classification/test.csv')

submittionDF['pathes']=submittionDF.apply(lambda x: constructpath(x,train=False),axis=1)

submittionDF['celltype']=submittionDF['experiment'].apply(lambda x: x.split('-')[0])



idxHEPG2test=submittionDF[submittionDF['celltype']=='HEPG2'].index

idxHUVECtest=submittionDF[submittionDF['celltype']=='HUVEC'].index

idxRPEtest=submittionDF[submittionDF['celltype']=='RPE'].index

idxU2OStest=submittionDF[submittionDF['celltype']=='U2OS'].index





#change INPUT SHAPE from (224,224) to (512,512)



basemodel=keras.applications.xception.Xception(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1108)



'''

global graph

graph = tf.get_default_graph() 



#EXTRACT FEATURES from pictures using model it outputs some logs with time remain to the feature extraction finish



traincontrols1=featurecollector(DF_train_control,basemodel, weights,BATCHSIZE,site=0)

traincontrols2=featurecollector(DF_train_control,basemodel, weights,BATCHSIZE,site=1)

traincontrols=(np.mean([traincontrols1[0],traincontrols2[0]],axis=0), traincontrols1[1])

MODELINPS1=featurecollector(DF,basemodel, weights,BATCHSIZE,site=0)

#pickle.dump(MODELINPS1, open("firstsitesfeatures","wb"))

MODELINPS2=featurecollector(DF,basemodel, weights,BATCHSIZE,site=1)

#pickle.dump(MODELINPS2, open("secondsitesfeatures","wb"))

# calculate difference in features compared to control

MODELINPS_1=np.zeros((MODELINPS1[0].shape))

MODELINPS_2=np.zeros((MODELINPS2[0].shape))

for i,row in DF.iterrows():

    ind = DF_train_control[DF_train_control['experiment']==row['experiment']][DF_train_control['plate']==row['plate']][DF_train_control['well_type']=='negative_control'].index[0]

    MODELINPS_1[i,:]=MODELINPS1[0][i,:]-traincontrols[0][ind,:]

    MODELINPS_2[i,:]=MODELINPS2[0][i,:]-traincontrols[0][ind,:]

    

#reading and preprocessing the test data, creating test features on site 0 and site 1 





featurestest1=featurecollector(submittionDF,basemodel,weights,BATCHSIZE,site=0,test=True)

featurestest2=featurecollector(submittionDF,basemodel,weights,BATCHSIZE,site=1,test=True)

testcontrols1=featurecollector(DF_test_control,basemodel, weights,BATCHSIZE,site=0)

testcontrols2=featurecollector(DF_test_control,basemodel, weights,BATCHSIZE,site=1)

featurestest_1=np.zeros((featurestest1[0].shape))

featurestest_2=np.zeros((featurestest2[0].shape))

testcontrols=(np.mean([testcontrols1[0],testcontrols2[0]],axis=0),testcontrols1[1])

for i,row in submittionDF.iterrows():

    ind = DF_test_control[DF_test_control['experiment']==row['experiment']][DF_test_control['plate']==row['plate']][DF_test_control['well_type']=='negative_control'].index[0]

    featurestest_1[i,:]=featurestest1[0][i,:]-testcontrols[0][ind,:]

    featurestest_2[i,:]=featurestest2[0][i,:]-testcontrols[0][ind,:]

    

Xsez=np.concatenate([MODELINPS_1,MODELINPS_2],axis=0)

Ysez1=np.concatenate([MODELINPS1[1],MODELINPS2[1]],axis=0)

pickle.dump(Xsez, open("Xsez","wb"))

pickle.dump(Ysez1, open("Ysez1","wb"))



pickle.dump(featurestest_1, open('Xtest1','wb'))

pickle.dump(featurestest_2, open('Xtest2','wb'))



'''

os.listdir('../input/xception-cells-features-extracted/')
Xsez=pickle.load(open('../input/xception-cells-features-extracted/Xsez','rb'))

Ysez1=pickle.load(open('../input/xception-cells-features-extracted/Ysez1','rb'))

Xtest1=pickle.load(open('../input/xception-cells-features-extracted/Xtest1','rb'))

Xtest2=pickle.load(open('../input/xception-cells-features-extracted/Xtest2','rb'))
Ysez=keras.utils.to_categorical(Ysez1, num_classes=1108)
#Here and in Createmodel() function you can play with classification model

models={}

for modelname in ['modelHEPG2','modelHUVEC','modelRPE','modelU2OS']:

    models[modelname]=[

    Createmodel(basemodel,layers=[128,128],reg=0.01,lr=0.001,regllast=0.001,arcface=False),

    Createmodel(basemodel,layers=[128,128],reg=0.01,lr=0.001,regllast=0.001,arcface=False),

    Createmodel(basemodel,layers=[128,128],reg=0.01,lr=0.001,regllast=0.001,arcface=False),

    Createmodel(basemodel,layers=[128,128],reg=0.01,lr=0.001,regllast=0.001,arcface=False),

    ]

'''modelHEPG2=Createmodel(basemodel,layers=[128,128],reg=0.01,lr=0.001,regllast=0.001,arcface=False)

modelHUVEC=Createmodel(basemodel,layers=[128,128],reg=0.01,lr=0.001,regllast=0.001,arcface=False)

modelRPE=Createmodel(basemodel,layers=[128,128],reg=0.01,lr=0.001,regllast=0.001,arcface=False)

modelU2OS=Createmodel(basemodel,layers=[128,128],reg=0.01,lr=0.001,regllast=0.001,arcface=False)'''
#models['modelHEPG2'][0].summary()
np.random.seed(123)

np.random.shuffle(idxHEPG2)

np.random.shuffle(idxHUVEC)

np.random.shuffle(idxRPE)

np.random.shuffle(idxU2OS)

indicies={

    'modelHEPG2':idxHEPG2,

    'modelHUVEC':idxHUVEC,

    'modelRPE':idxRPE,

    'modelU2OS':idxU2OS

}
kf=sklearn.model_selection.KFold(n_splits=4)

hist={}

for modelname in ['modelHEPG2','modelHUVEC','modelRPE','modelU2OS']: 

    hist[modelname]=[0,1,2,3]

for modelname in ['modelHEPG2','modelHUVEC','modelRPE','modelU2OS']:

    i=0

    for train_index, val_index in kf.split(indicies[modelname]):

        hist[modelname][i]=models[modelname][i].fit(

            Xsez[train_index],Ysez[train_index],

            epochs=50, 

            batch_size=BATCHSIZE,

            validation_data=(Xsez[val_index],Ysez[val_index]),

            callbacks=[keras.callbacks.ModelCheckpoint(modelname+str(i)+'.hdf5', monitor='val_categorical_accuracy', verbose=0, save_best_only=True),keras.callbacks.LearningRateScheduler(schedule)]

        )

        i=i+1
for modelname in ['modelHEPG2','modelHUVEC','modelRPE','modelU2OS']:

    for i in range(4):

        models[modelname][i]=keras.models.load_model(modelname+str(i)+'.hdf5')

#np.max(histHEPG2.history['val_categorical_accuracy'])

#np.max(histHEPG2.history['categorical_accuracy'])
#make predictions on site 0 and 1

answer1=np.zeros((len(Xtest1),1108))

answer2=np.zeros((len(Xtest2),1108))

for ind, modelname in zip([idxHEPG2test,idxHUVECtest,idxRPEtest,idxU2OStest],['modelHEPG2','modelHUVEC','modelRPE','modelU2OS']):

    for i in range(4):

        answer1[ind]=answer1[ind]+models[modelname][i].predict(Xtest1[ind],verbose=1)

        answer2[ind]=answer2[ind]+models[modelname][i].predict(Xtest2[ind],verbose=1)
#Create submission file

submittionDF['sirna']=np.argmax(answer1+answer2, axis=1)

submittionDF.to_csv('submission.csv', sep=',', columns=['id_code', 'sirna'], header=True, index=False)